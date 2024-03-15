import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
from cliport.models.core.attention import Attention
from cliport.models.core.transport import Transport
from cliport.models.streams.two_stream_attention import TwoStreamAttention
from cliport.models.streams.two_stream_transport import TwoStreamTransport

from cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from cliport.models.streams.two_stream_transport import TwoStreamTransportLat

from uncertainty_module.src.base.calib_scaling import CalibScaler
from uncertainty_module.src.temperature_scaling.temperature_scaling import TemperatureScaler
from uncertainty_module.action_selection import ActionSelection

class TransporterAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)
    
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']

        if self.cfg['calibration']['enabled']:
            self.save_steps = cfg['calibration']['save_steps']
        else:
            self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }

        if self.cfg['calibration']['enabled']:
            # self._optimizers = None
            
            if self.cfg['calibration']['calib_type'] == 'temperature':
                self.calib_scaler = TemperatureScaler(device=self.device_type, cfg=self.cfg)
            else:
                self.calib_scaler = CalibScaler(device=self.device_type, cfg=self.cfg)
            if self.cfg['calibration']['training']:
                self.train_ds = test_ds # in training, test_ds is the validation set, set it to validation set in calibration

        if self.cfg['action_selection']['enabled']:
            self.action_selection = ActionSelection(device=self.device_type, 
                                                    batch_size=1,
                                                    enabled=self.cfg['action_selection']['enabled'],
                                                    attn_tau=self.cfg['action_selection']['attn_tau'],
                                                    trans_tau=self.cfg['action_selection']['trans_tau'],
                                                    attn_uaa=self.cfg['action_selection']['attn_uaa'],
                                                    trans_uaa=self.cfg['action_selection']['trans_uaa']
                                                    )
        
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))
        
    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        if backprop:
            transport_optim = self._optimizers['trans']
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
                'theta': np.absolute((theta - p1_theta) % np.pi)
            }
        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        if self.cfg['calibration']['enabled']:

            # import pdb; pdb.set_trace()
            # # Get training losses
            # self.attention.eval()
            # self.transport.eval()
            
            frame, _ = batch
            
            # get calibration loss:
            step = self.total_steps + 1
            
            img = frame['img']
            lang_goal = frame['lang_goal']
            p0 = frame['p0']
            p1, p1_theta = frame['p1'], frame['p1_theta']

            # Attention model forward pass.
            pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
            pick_conf = self.attn_forward(pick_inp, softmax=False)
            # scaling attention layer

            pick_conf = self.calib_scaler.scale_attn(pick_conf)
            # compute attention loss
            loss0, _err0 = self.calib_scaler.calib_attn_criterion(backprop=self.calib_scaler.training,
                                                compute_err=True,
                                                inp=pick_inp,
                                                out=pick_conf,
                                                p=frame['p0'],
                                                theta=frame['p0_theta'])
            
            # backpropagate calibration for attention
            if self.calib_scaler.training:
                calib_attn_optim = self.calib_scaler._optimizers['attn_calib']
                self.manual_backward(loss0, calib_attn_optim)
                calib_attn_optim.step()
                calib_attn_optim.zero_grad()
                
                self.calib_scaler.attn_scheduler.step()

            # Transport model forward pass.
            place_inp = {'inp_img': img, 'p0': p1, 'lang_goal': lang_goal}
            place_conf = self.trans_forward(place_inp, softmax=False)
            # scaling transporter layer
            place_conf = self.calib_scaler.scale_trans(place_conf)

            # compute transporter loss
            loss1, _err1 = self.calib_scaler.calib_transport_criterion(backprop=self.calib_scaler.training,
                                                compute_err=True,
                                                inp=place_inp,
                                                output=place_conf,
                                                p=frame['p0'],
                                                q=frame['p1'],
                                                theta=frame['p1_theta'])
            # import pdb; pdb.set_trace()   
            # backpropagate calibration for transporter
            if self.calib_scaler.training:
                transport_optim = self.calib_scaler._optimizers['trans_calib']
                self.manual_backward(loss1, transport_optim)
                transport_optim.step()
                transport_optim.zero_grad()
                
                self.calib_scaler.trans_scheduler.step()

            total_loss = loss0 + loss1
            # import pdb; pdb.set_trace()           
            self.log('calib/attn/attn_temperature', self.calib_scaler.attn_temperature)
            self.log('calib/trans/trans_temperature', self.calib_scaler.trans_temperature)
            self.log('calib/attn/lr', self.calib_scaler._optimizers['attn_calib'].param_groups[0]['lr'])
            self.log('calib/trans/lr', self.calib_scaler._optimizers['trans_calib'].param_groups[0]['lr'])
            self.log('calib/attn/loss', loss0)
            self.log('calib/trans/loss', loss1)
            self.log('calib/loss', total_loss)
            self.log('tr/attn/loss', loss0)
            self.log('tr/trans/loss', loss1)
            self.log('tr/loss', total_loss)
            self.total_steps = step

            self.trainer.train_loop.running_loss.append(total_loss)

            self.check_save_iteration()

            return dict(
                loss=total_loss,
            )
            
        else:
            self.attention.train()
            self.transport.train()

            frame, _ = batch

            # Get training losses.
            step = self.total_steps + 1
            loss0, err0 = self.attn_training_step(frame)
            if isinstance(self.transport, Attention):
                loss1, err1 = self.attn_training_step(frame)
            else:
                loss1, err1 = self.transport_training_step(frame)
            total_loss = loss0 + loss1
            
            self.log('tr/attn/loss', loss0)
            self.log('tr/trans/loss', loss1)
            self.log('tr/loss', total_loss)
            self.total_steps = step

            self.trainer.train_loop.running_loss.append(total_loss)

            self.check_save_iteration()

            return dict(
                loss=total_loss,
            )

    # def on_save_checkpoint(self, checkpoint):
    #     if self.cfg['calibration']['enabled']:
    #         calib_scaler_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('calib_scaler.')]
    #         calib_scaler_state = {k: checkpoint['state_dict'].pop(k) for k in calib_scaler_keys}
    #         if calib_scaler_state:
    #             self.calib_scaler.save_parameter(self.trainer.current_epoch, self.trainer.global_step)
    #     return checkpoint
    def on_train_start(self):
        if self.cfg['calibration']['enabled']:
            print('logging temperature')
            print(f'attn_temperature: {self.calib_scaler.attn_temperature}, \n trans_temperature: {self.calib_scaler.trans_temperature}')
            wandb_run = self.logger.experiment
            wandb_run.log({
                'calib/attn/attn_temperature': self.calib_scaler.attn_temperature.item(),
                'calib/trans/trans_temperature': self.calib_scaler.trans_temperature.item()
            }, step=0, commit=True)

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            # import pdb; pdb.set_trace()
            if not self.cfg['calibration']['enabled']:
                self.trainer.run_evaluation()
                val_loss = self.trainer.callback_metrics['val_loss']
                steps = f'{global_step + 1:05d}'
                filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
                checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
                ckpt_path = os.path.join(checkpoint_path, filename)
                self.trainer.save_checkpoint(ckpt_path)
            else:
                # return
                # val_loss = self.trainer.callback_metrics['calib/loss']
                # steps = f'{global_step + 1:05d}'
                # filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
                # checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
                # ckpt_path = os.path.join(checkpoint_path, filename)
                # self.trainer.save_checkpoint(ckpt_path)
                if self.cfg['calibration']['training']:
                    # calib_scaler_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('calib_scaler.')]
                    # calib_scaler_state = {k: checkpoint['state_dict'].pop(k) for k in calib_scaler_keys}
                    # if calib_scaler_state:
                    self.calib_scaler.save_parameter(self.trainer.current_epoch, self.trainer.global_step)
                return
                        
        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            if self.cfg['calibration']['training']:
                pass
            else:
                self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch+1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        
        if self.cfg['action_selection']['enabled']:
            # Get heightmap from RGB-D images.
            img = self.test_ds.get_image(obs)

            # Attention model forward pass.
            pick_inp = {'inp_img': img}
            
            if self.cfg['calibration']['enabled']:
                output = self.attn_forward(pick_inp, softmax=False)
                output = self.calib_scaler.scale_attn(output)
                output = F.softmax(output, dim=-1)
                
                in_shape = self.in_shape
                pick_conf = output.reshape([in_shape[0], in_shape[1], 1])
            else:
                pick_conf = self.attn_forward(pick_inp)
            if self.cfg['action_selection']['attn_uaa']:
                pick_conf = self.action_selection.get_uncertainty_heatmap(pick_conf)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            # Transport model forward pass.
            place_inp = {'inp_img': img, 'p0': p0_pix}
            if self.cfg['calibration']['enabled']:
                output = self.trans_forward(place_inp, softmax=False)
                output_shape = output.shape
                output = output.reshape((1, np.prod(output.shape)))
                output = self.calib_scaler.scale_trans(output)
                output = F.softmax(output, dim=-1) # add scaling
                place_conf = output.reshape(output_shape[1:])
            else:
                place_conf = self.trans_forward(place_inp)
            if self.cfg['action_selection']['trans_uaa']:
                place_conf = self.action_selection.get_uncertainty_heatmap(place_conf)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])            
        
        else:
            # Get heightmap from RGB-D images.
            img = self.test_ds.get_image(obs)

            # Attention model forward pass.
            pick_inp = {'inp_img': img}
            pick_conf = self.attn_forward(pick_inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            # Transport model forward pass.
            place_inp = {'inp_img': img, 'p0': p0_pix}
            place_conf = self.trans_forward(place_inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass
        # if self.calib_scaler and self.calib_scaler.training:
        #     return list(self.calib_scaler._optimizers.values())
        # else:
        #     return None

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds
    
    def test_dataloader(self):
        return self.train_ds ## In test mode, we use the train_ds as the test_ds

    def load(self, model_path):
        if self.cfg['calibration']['use_hard_temp']:
            params = torch.load(model_path)['state_dict']
            # params.pop('calib_scaler.attn_temperature')
            # params.pop('calib_scaler.trans_temperature')
            params['calib_scaler.attn_temperature'] = torch.nn.Parameter(torch.ones(1, device=self.device))
            params['calib_scaler.trans_temperature'] = torch.nn.Parameter(torch.ones(1, device=self.device))
            self.load_state_dict(params)
            pass
        else:
            self.load_state_dict(torch.load(model_path)['state_dict'])
            if self.cfg['calibration']['enabled']:
                self.calib_scaler.load_parameter()

        self.to(device=self.device_type)
        

    
    def test_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        #TODO: figure out attn_forward(inp, softmax=False) and attn_forward(inp)
        frame, _ = batch
        
        # get calibration loss:
        step = self.total_steps + 1
        
        img = frame['img']
        lang_goal = frame['lang_goal']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}

        # eval if pred is correct
        err0 = self.calib_scaler.calib_attn_eval(attn_forward=self.attn_forward, 
                                                 in_shape = self.in_shape,
                                                 inp=pick_inp, 
                                                 p=frame['p0'], 
                                                 theta=frame['p0_theta'])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p1, 'lang_goal': lang_goal}
        
        # compute transporter loss
        err1 = self.calib_scaler.calib_trans_eval(trans_forward=self.trans_forward, 
                                                  in_shape = self.in_shape,
                                                  inp=place_inp, 
                                                  q=frame['p1'], 
                                                  theta=frame['p1_theta'])
        

        # self.check_save_iteration()
        # print(f'err0{err0}, err1{err1}')
        # if err0['dist']==0 or err1['dist']==0:
        #     import pdb; pdb.set_trace()
        
        # self.log('test/attn_err', err0['dist'])
        # self.log('test/trans_err', err1['dist'])

        return dict(
            err0=err0,
            err1=err1
        )
    
    def test_epoch_end(self, all_outputs):
        return all_outputs
        # self.calib_scaler.attn_err

class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class ClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_unet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
                
class TwoStreamClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetLatTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_unet_lat'
        self.attention = TwoStreamAttentionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipWithoutSkipsTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_woskip'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
