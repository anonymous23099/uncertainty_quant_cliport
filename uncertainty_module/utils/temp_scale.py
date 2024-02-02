import torch
import torch.nn as nn
import torch.nn.functional as F


rotation_resolution = 5
num_rotation_classes = int(360. // rotation_resolution)
replay_batch_size = 8

def softmax_q_trans(self, q):
    q_shape = q.shape
    return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

def softmax_q_rot_grip(self, q_rot_grip):
    q_rot_x_flat = q_rot_grip[:, 0*num_rotation_classes: 1*num_rotation_classes]
    q_rot_y_flat = q_rot_grip[:, 1*num_rotation_classes: 2*num_rotation_classes]
    q_rot_z_flat = q_rot_grip[:, 2*num_rotation_classes: 3*num_rotation_classes]
    q_grip_flat  = q_rot_grip[:, 3*num_rotation_classes:]

    q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
    q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
    q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
    q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

    return torch.cat([q_rot_x_flat_softmax,
                        q_rot_y_flat_softmax,
                        q_rot_z_flat_softmax,
                        q_grip_flat_softmax], dim=1)
    
def softmax_ignore_collision(self, q_collision):
    q_collision_softmax = F.softmax(q_collision, dim=1)
    return q_collision_softmax


def _argmax_3d(tensor_orig):
    b, c, d, h, w = tensor_orig.shape  # c will be one
    idxs = tensor_orig.view(b, c, -1).argmax(-1)
    indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
    return indices

def choose_highest_action(q_trans, q_rot_grip, q_collision):
    coords = _argmax_3d(q_trans)
    rot_and_grip_indicies = None
    ignore_collision = None
    if q_rot_grip is not None:
        q_rot = torch.stack(torch.split(
            q_rot_grip[:, :-2],
            int(360 // rotation_resolution),
            dim=1), dim=1)
        rot_and_grip_indicies = torch.cat(
            [q_rot[:, 0:1].argmax(-1),
                q_rot[:, 1:2].argmax(-1),
                q_rot[:, 2:3].argmax(-1),
                q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
        ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
    return coords, rot_and_grip_indicies, ignore_collision