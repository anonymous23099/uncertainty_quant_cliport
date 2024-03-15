import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler
import numpy as np

from uncertainty_module.src.base.calib_scaling import CalibScaler

class TemperatureScaler(CalibScaler):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        self.to(device=device)
        self.cfg = cfg
        self.training = self.cfg['calibration']['training']
        self.training_iter = self.cfg['calibration']['training_iter']
        self.scaler_log_root = self.cfg['calibration']['scaler_log_root']
        self.calib_type = self.cfg['calibration']['calib_type']

        self.calib_attn_lr = self.cfg['calibration']['calib_attn_lr']
        self.calib_trans_lr = self.cfg['calibration']['calib_trans_lr']
        
        self.use_hard_temp = self.cfg['calibration']['use_hard_temp']

        self.attn_n_rotations = 1
        self.trans_n_rotations = self.cfg['train']['n_rotations']
        self.n_rotations = self.trans_n_rotations
        
        self.n_demos = cfg['calibration']['n_demos']
        self.step_size = self.n_demos * 3 # 20 # update lr per 10 epochs
        self.scaling_in_eval = cfg['calibration']['scaling_in_eval']
        
        print('temp logging dir:', self.scaler_log_root)
        # breakpoint()
        if self.use_hard_temp:
            print('using hard temp')
            # self.training = False
            self.attn_temperature = torch.nn.Parameter(torch.ones(1, device=self.device))
            self.trans_temperature = torch.nn.Parameter(torch.ones(1, device=self.device))
        else:
            self.attn_temperature = torch.nn.Parameter(torch.ones(1, device=self.device))
            self.trans_temperature = torch.nn.Parameter(torch.ones(1, device=self.device))

        if self.training:
            self._optimizers = {
                'attn_calib': torch.optim.Adam([self.attn_temperature], lr=self.calib_attn_lr),
                'trans_calib': torch.optim.Adam([self.trans_temperature], lr=self.calib_trans_lr)
                }
            self.attn_scheduler = lr_scheduler.StepLR(self._optimizers['attn_calib'], 
                                            step_size=self.step_size, 
                                            gamma=0.5)
            self.trans_scheduler = lr_scheduler.StepLR(self._optimizers['trans_calib'], 
                                            step_size=self.step_size, 
                                            gamma=0.5)
        else:
            self._optimizers = None

        self.file_name = self.cfg['train']['task']+'_calib_params.pth'
        # load the temperature of unseen task with seen task
        self.file_name = self.file_name.replace('unseen', 'seen')
        print('file_name', self.file_name)
        # self.saving_dir = os.path.join(self.cfg['calibration']['scaler_log_root'], )
        self.saving_dir = os.path.join(self.cfg['train']['train_dir'], self.file_name)
        # if not self.training:
        #     self.attn_err = []
        #     self.trans_err = []
        #     self.pick_conf_max_list = []
        #     self.place_conf_max_list = []

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def calib_attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attn_n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attn_n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attn_n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)
        # import pdb; pdb.set_trace()
        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # # Backpropagate.
        # if backprop:
        #     attn_optim = self._optimizers['attn_calib']
        #     self.manual_backward(loss, attn_optim)
        #     attn_optim.step()
        #     attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        # if compute_err:
        #     pick_conf = self.attn_forward(inp)
        #     pick_conf = pick_conf.detach().cpu().numpy()
        #     argmax = np.argmax(pick_conf)
        #     argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        #     p0_pix = argmax[:2]
        #     p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        #     err = {
        #         'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
        #         'theta': np.absolute((theta - p0_theta) % np.pi)
        #     }
        return loss, err
    
    def calib_transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        itheta = theta / (2 * np.pi / self.trans_n_rotations)
        itheta = np.int32(np.round(itheta)) % self.trans_n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.trans_n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        # if backprop:
        #     transport_optim = self._optimizers['trans_calib']
        #     self.manual_backward(loss, transport_optim)
        #     transport_optim.step()
        #     transport_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        # if compute_err:
        #     place_conf = self.trans_forward(inp)
        #     place_conf = place_conf.permute(1, 2, 0)
        #     place_conf = place_conf.detach().cpu().numpy()
        #     argmax = np.argmax(place_conf)
        #     argmax = np.unravel_index(argmax, shape=place_conf.shape)
        #     p1_pix = argmax[:2]
        #     p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        #     err = {
        #         'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
        #         'theta': np.absolute((theta - p1_theta) % np.pi)
        #     }

        # self.transport.iters += 1
        return loss, err

    def scale_attn(self, logits):
        return logits/self.attn_temperature
    
    def scale_trans(self, logits):
        return logits/self.trans_temperature

    def scale_logits(self, logits):
        if self.temperature is None:
            raise ValueError("Temperature not set. First run the calibration process.")
        print('hard_temp', self.temperature)
        q_trans, q_rot_grip, q_collision = logits
        return [q_trans/self.temperature, q_rot_grip/self.temperature, q_collision/self.temperature]
    
    def get_val(self):
        return self.temperature

    def calib_attn_eval(self, attn_forward, in_shape, inp, p, theta):
        # pick_conf = attn_forward(inp)
        #TODO: add scaling
        output = attn_forward(inp, softmax=False)
        if self.scaling_in_eval:
            output = self.scale_attn(output)
        output = F.softmax(output, dim=-1)
        pick_conf = output.reshape([in_shape[0], in_shape[1], 1])

        pick_conf = pick_conf.detach().cpu().numpy()

        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        pick_up_max = pick_conf[argmax]
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
        
        import matplotlib.pyplot as plt

        # Select the first three channels
        rgb_image = inp['inp_img'][:, :, :3]

        # Ensure the values are in the range [0, 1] (matplotlib requirement for RGB images)
        rgb_image = rgb_image / np.max(rgb_image)
        # Save the image
        plt.imsave('/home/bobwu/cliport/shared/rgb_image.png', rgb_image.transpose(1,0,2))
        # # Visualize pick_conf as a heatmap
        # import matplotlib.pyplot as plt
        # plt.imshow(pick_conf.transpose(1,0,2), cmap='hot', interpolation='nearest')
        
        # # Add p to the image
        # p_np = np.round(np.array(p[::-1]), 2)  # Reverse the array before rounding
        # plt.plot(p_np[1], p_np[0], 'g+', markersize=10, markeredgewidth=2)#, markeredgewidth=1, markeredgecolor='k')
        # p0_pix_plot = p0_pix[::-1]
        # plt.plot(p0_pix_plot[1], p0_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)
        

        # plt.colorbar()
        # plt.savefig('/home/bobwu/cliport/shared/pick_conf.png')
        # plt.close()
        # import pdb; pdb.set_trace()

        err = {
            'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
            'theta': np.absolute((theta - p0_theta) % np.pi),
            'pick_up_max': pick_up_max
        }
        return err
    
    def calib_trans_eval(self, trans_forward, in_shape, inp, q, theta):
        # place_conf = trans_forward(inp)
        # import pdb; pdb.set_trace()
       
        output = trans_forward(inp, softmax=False)
        output_shape = output.shape
        output = output.reshape((1, np.prod(output.shape)))
        if self.scaling_in_eval:
            output = self.scale_trans(output)
        output = F.softmax(output, dim=-1) # add scaling
        place_conf = output.reshape(output_shape[1:])
        
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        place_conf_max = place_conf[argmax]
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
        # import pdb; pdb.set_trace()

        # Visualize pick_conf as a heatmap
        import matplotlib.pyplot as plt
        place_conf_plot = place_conf[:,:,argmax[2]][:, :, np.newaxis] # (320, 160, 1)
        plt.imshow(place_conf_plot.transpose(1,0,2), cmap='hot', interpolation='nearest')
        
        # Add p to the image
        q_np = np.round(np.array(q[::-1]), 2)  # Reverse the array before rounding
        plt.plot(q_np[1], q_np[0], 'g+', markersize=10, markeredgewidth=2)#, markeredgewidth=1, markeredgecolor='k')
        p1_pix_plot = p1_pix[::-1]
        plt.plot(p1_pix_plot[1], p1_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)
        

        plt.colorbar()
        plt.savefig('/home/bobwu/cliport/shared/place_conf.png')
        plt.close()
        # import pdb; pdb.set_trace()

        err = {
            'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
            'theta': np.absolute((theta - p1_theta) % np.pi),
            'place_conf_max': place_conf_max
        }
        return err

    def save_parameter(self, epoch, global_step):
        # torch.save(self.state_dict(), self.saving_dir)
        state = {
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': self.state_dict()
            }
        print('saving global_step', global_step)
        print('saving epoch', epoch)
        torch.save(state, self.saving_dir)

    def load_parameter(self):
        if os.path.exists(self.saving_dir):
            try:
                state = torch.load(self.saving_dir)
                self.load_state_dict(state['state_dict'])
                self._current_epoch = state['epoch']
                self._global_step = state['global_step']
                print('loading global_step', self._global_step)
                print('loading epoch', self._current_epoch)
                print('loaded temperature. attn: {}, trans: {}'.format(self.attn_temperature, self.trans_temperature))
            except Exception as e:
                print(f"Error loading parameters: {e}")
        else:
            self._current_epoch = 0
            self._global_step = 0
            print(f"No parameters found at {self.saving_dir}")
            if self.training:
                print('Initializing attn_temperature and trans_temperature to 1.0')
            else:
                exit()
    def get_current_epoch(self):
        return self._current_epoch

    def get_global_step(self):
        return self._global_step
        
    def configure_optimizers(self):
        pass

    # def save_parameter(self, task_name=None, savedir=None):
    #     savedir = self.scaler_log_root
        
    #     if not task_name:
    #         temp_file = "temperature.pth"
    #     else:
    #         savedir = os.path.join(savedir, task_name)
    #         temp_file = task_name + "_temperature.pth"
    #     full_path = os.path.join(savedir, temp_file)

    #     if not os.path.exists(savedir):
    #         os.makedirs(savedir)
    #     torch.save(self.temperature, full_path)
    
    # def load_parameter(self, task_name=None, savedir=None):
    #     savedir = self.scaler_log_root
    #     # if using hard temp, don't load
    #     if not self.use_hard_temp:
    #         if not task_name:
    #             temp_file = "temperature.pth"
    #         else:
    #             savedir = os.path.join(savedir, task_name)
    #             temp_file = task_name + "_temperature.pth"
    #         full_path = os.path.join(savedir, temp_file)
                
    #         if os.path.exists(full_path):
    #             loaded_temperature = torch.load(full_path)
    #             self.temperature.data = loaded_temperature.data
    #         # TODO: if it does not exist, don't load it
    #         else:
    #             print(f"Error: No weights found at {full_path}")
    #             print("Initializing temperature to 1.0")
    #             self.temperature.data.fill_(1.0)
    #     else:
    #         # print("Initializing temperature to hard coded temperature")
    #         # self.temperature.data.fill_(self.temperature)
    #         print("Using hardcoded temperature:", self.temperature)
    #         pass
