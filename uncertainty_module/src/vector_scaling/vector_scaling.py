import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler

# from utils.temp_scale import softmax_q_trans, softmax_q_rot_grip, softmax_ignore_collision, choose_highest_action

class LinearTransform(nn.Module):
    # def __init__(self, trans_dim, batch_size):
    #     super(LinearTransform, self).__init__()
    #     #TODO: change this for broadcasting
    #     self.trans_w = torch.nn.Parameter(torch.ones(batch_size, trans_dim**3))
    #     self.trans_b = torch.nn.Parameter(torch.zeros(batch_size, trans_dim**3))
        
    # def forward(self, x):
    #     x_1d = x.view(x.shape[0], -1)  # reshape with batch size as the 0-th dimension
    #     out_2d = self.trans_w * x_1d + self.trans_b
    #     out = out_2d.view(x.shape)
    #     return out
    
    def __init__(self, trans_dim, batch_size):
        super(LinearTransform, self).__init__()
        self.batch_size = batch_size
        self.trans_w = torch.nn.Parameter(torch.ones(trans_dim**3))
        self.trans_b = torch.nn.Parameter(torch.zeros(trans_dim**3))
        
    def forward(self, x):
        
        # Broadcasting
        w_broadcasted = self.trans_w.unsqueeze(0).expand(self.batch_size, -1)
        b_broadcasted = self.trans_b.unsqueeze(0).expand(self.batch_size, -1)
        
        x_1d = x.view(self.batch_size, -1)
        out_2d = w_broadcasted * x_1d + b_broadcasted
        out = out_2d.view(x.shape)
        return out

class VectorScaler:
    def __init__(self, calib_type,
                 device, 
                 rotation_resolution, 
                 batch_size, 
                 num_rotation_classes, 
                 voxel_size,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 training: bool = False,
                 training_iter: int = 10000,
                 scaler_log_root: str = None,
                 div_penalty: int = 0,
                 learning_rate: float = 1e-2,
                 ):
        self.calib_type = calib_type
        self.device = device
        self.training = training
        self.training_iter = training_iter
        self.scaler_log_root = scaler_log_root
        self.div_penalty = div_penalty
        self.lr = learning_rate
        print('scaler training logging dir:', self.scaler_log_root)
        print('vector_training_iter',training_iter)
        # if not use_hard_temp:
        #     self.temperature = torch.nn.Parameter(torch.ones([1,0], device=self.device))
        # else:
        #     self.training = False
        #     self.temperature = hard_temp * torch.nn.Parameter(torch.ones(1, device=self.device))
        
        # self.param = torch.nn.Parameter(torch.ones([1,0], device=self.device))

        self.trans_dim = 100
        self.model = LinearTransform(self.trans_dim, batch_size).to(device=self.device)
        
        
        if self.training:
            def lr_lambda(epoch):
                return 1 / ((epoch + 1) ** 0.5) 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
            # self.scheduler = lr_scheduler.StepLR(self.optimizer, 
            #                                 step_size=self.training_iter//50,  # 20 seems to be a good value
            #                                 gamma=0.1)
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            AssertionError('Need to load weights')
            self.optimizer = None

        self._rotation_resolution = rotation_resolution
        self._batch_size = batch_size
        self.bs = batch_size
        self._num_rotation_classes = num_rotation_classes
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._voxel_size = voxel_size
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        
        # one-hot zero tensors
        self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                        1,
                                                        self._voxel_size,
                                                        self._voxel_size,
                                                        self._voxel_size),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                        2),
                                                        dtype=int,
                                                        device=device)
        self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                    2),
                                                                    dtype=int,
                                                                    device=device)

        
    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision


    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def compute_loss(self, logits, labels):
        
        q_trans, q_rot_grip, q_collision = logits 
        
        action_trans, action_rot_grip, action_ignore_collisions = labels
        
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self.choose_highest_action(q_trans, q_rot_grip, q_collision)
        
        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(self.bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat = q_trans.view(self.bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(self.bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(self.bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat =  q_rot_grip[:, 3*self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()
        penalty = self.div_penalty * (torch.sum(self.model.trans_w-1)**2 + torch.sum(self.model.trans_b**2)) / self.trans_dim**3 /self.bs
        # print('non scaled penalty', penalty.item()/self.div_penalty)
        # print('total loss', total_loss.item())
        # print('weight mean', torch.sum(self.model.trans_w).item()/100**3/self.bs)
        # print('weight min max', torch.min(self.model.trans_w).item(), torch.max(self.model.trans_w).item())
        self.info = {}
        
        self.info['non_scaled_penalty'] = penalty.item()/self.div_penalty if self.div_penalty!=0 else penalty.item()
        self.info['total_loss'] = total_loss.item()
        self.info['weight_mean'] = torch.sum(self.model.trans_w).item() / (100**3)
        self.info['weight_min'] = torch.min(self.model.trans_w).item()
        self.info['weight_max'] = torch.max(self.model.trans_w).item()
        self.info['bias_mean'] = torch.sum(self.model.trans_b).item() / (100**3)
        self.info['bias_min'] = torch.min(self.model.trans_b).item()
        self.info['bias_max'] = torch.max(self.model.trans_b).item()
        
        return total_loss + penalty# q_trans_loss
            
    def scale_logits(self, logits):
        if self.model is None:
            raise ValueError("Parameters not set. First run the calibration process.")
        q_trans, q_rot_grip, q_collision = logits
        # print(q_trans.shape, q_rot_grip.shape, q_collision.shape)
        # exit()
        # print('---')
        q_trans_transformed = self.model(q_trans)
        return [q_trans_transformed, q_rot_grip, q_collision]

    def get_calibrated_probs(self, logits):
        scaled_logits = self.scale_logits(logits)
        [q_trans, q_rot_grip, q_collision] = scaled_logits
        
        trans_conf = self._softmax_q_trans(q_trans).max().detach().cpu().item()
        rot_grip_conf = self._softmax_q_rot_grip(q_rot_grip)
        rot_conf = torch.stack(torch.split(rot_grip_conf[:, :-2],int(360 // self._rotation_resolution),dim=1), dim=1)
        rot_conf = torch.prod(rot_conf.detach().cpu()[0].max(dim=1)[0]).item()

        grip_conf = rot_grip_conf[:, -2:].detach().cpu().max().item()
        collision_conf = self._softmax_ignore_collision(q_collision).max().detach().cpu().item()
        total_conf = trans_conf * rot_conf * grip_conf * collision_conf
        
        return total_conf

    def get_val(self):
        # return (torch.sum(self.model.trans_w-1)**2 + torch.sum(self.model.trans_b**2))/self.trans_dim**3
        return self.info
    
    def save_parameter(self, task_name=None, savedir=None):
        savedir = self.scaler_log_root
        
        if not task_name:
            temp_file = "vector_weights.pth"
        else:
            savedir = os.path.join(savedir, task_name)
            temp_file = task_name + "_vector_weights.pth"
        full_path = os.path.join(savedir, temp_file)
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save(self.model.state_dict(), full_path)
    
    def load_parameter(self, task_name=None, savedir=None):
        savedir = self.scaler_log_root
        print('task_name', task_name)
        if not task_name:
            temp_file = "vector_weights.pth"
        else:
            savedir = os.path.join(savedir, task_name)
            temp_file = task_name + "_vector_weights.pth"
        full_path = os.path.join(savedir, temp_file)

        if not os.path.exists(full_path):
            print(f"Error: No weights found at {full_path}")
            return
        self.model.load_state_dict(torch.load(full_path))
        
        if not self.training:
            # self.model.trans_w.data = self.model.trans_w.data.mean(dim=0, keepdim=True)
            # self.model.trans_b.data = self.model.trans_b.data.mean(dim=0, keepdim=True)
            # print('after averaging')
            # print(self.model.trans_w.data.shape)
            self.model.eval()  # Set the model to evaluation mode
