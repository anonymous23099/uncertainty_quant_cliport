from pytorch_lightning import LightningModule
import torch

class CalibScaler(LightningModule):
    def __init__(self, device, cfg):
        super().__init__()
        self.to(device=device)
        self.cfg = cfg
        self.training = self.cfg['calibration']['training']
        self.training_iter = self.cfg['calibration']['training_iter']
        self.scaler_log_root = self.cfg['calibration']['scaler_log_root']
        self.calib_type = self.cfg['calibration']['calib_type']

        self.calib_attn_lr = self.cfg['calibration']['calib_attn_lr']
        self.calib_trans_lr = self.cfg['calibration']['calib_trans_lr']


    
    def save_parameter(self, task_name=None):
        # Common method to save parameters
        pass
    
    def load(self):
        # Common method to load parameters
        pass
    
    def scale_logits(self, logits):
        pass
    
    def compute_loss(self, logits, labels):
        pass
    
    def configure_optimizers(self):
        pass