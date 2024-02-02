"""Main training script."""

import os
from pathlib import Path

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="./cfg", config_name='calib_train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg['tag']) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )


    # # Trainer
    # max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    # trainer = Trainer(
    #     gpus=cfg['train']['gpu'],
    #     fast_dev_run=cfg['debug'],
    #     logger=wandb_logger,
    #     checkpoint_callback=checkpoint_callback,
    #     max_epochs=max_epochs,
    #     automatic_optimization=False,
    #     check_val_every_n_epoch=max_epochs // 50,
    #     # resume_from_checkpoint=last_checkpoint,
    # )
    # # Resume epoch and global_steps
    # if last_checkpoint:
    #     print(f"Resuming: {last_checkpoint}")
    #     last_ckpt = torch.load(last_checkpoint)
    #     trainer.current_epoch = last_ckpt['epoch']
    #     trainer.global_step = last_ckpt['global_step']
    #     del last_ckpt

    # Update the Configs according to the need of calibration
    if cfg['calibration']['enabled']:
        cfg['dataset']['type'] = cfg['calibration']['datasets_type']

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    

    if cfg['calibration']['enabled']:
        # use validation for calibration
        # still provide val_ds as train_ds to avoid error
        agent = agents.names[agent_type](name, cfg, val_ds, val_ds) 
        
        for param in agent.attention.parameters():
            param.requires_grad = False
        for param in agent.transport.parameters():
            param.requires_grad = False
    else:
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
    
    # if last_checkpoint:
    #     print(f"Resuming: {last_checkpoint}")
    #     last_ckpt = torch.load(last_checkpoint)
    #     agent.load_state_dict(last_ckpt['state_dict'], strict=False)

    # Main training loop
    # Trainer
    if cfg['calibration']['enabled']:
        max_epochs = cfg['calibration']['n_steps'] // cfg['calibration']['n_demos']
        trainer = Trainer(
            gpus=cfg['train']['gpu'],
            fast_dev_run=cfg['debug'],
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            automatic_optimization=False,
            check_val_every_n_epoch=max_epochs // 200, #5,
            log_every_n_steps=1,
            limit_val_batches=0
            # resume_from_checkpoint=last_checkpoint,
        )
    else:
        max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
        trainer = Trainer(
            gpus=cfg['train']['gpu'],
            fast_dev_run=cfg['debug'],
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            automatic_optimization=False,
            check_val_every_n_epoch=max_epochs // 50,
            # resume_from_checkpoint=last_checkpoint,
        )
        
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint) # the checkpoint contains temperate parameters
        
        agent.load_state_dict(last_ckpt['state_dict'], strict=False)
        if cfg['calibration']['enabled'] and cfg['calibration']['training']:
            agent.calib_scaler.load_parameter()
            trainer.current_epoch = agent.calib_scaler.get_current_epoch()
            trainer.global_step = agent.calib_scaler.get_global_step()
        else:
            trainer.current_epoch = last_ckpt['epoch']
            trainer.global_step = last_ckpt['global_step']
        # import pdb; pdb.set_trace()
        del last_ckpt
        
    # import pdb; pdb.set_trace()
    trainer.fit(agent)

if __name__ == '__main__':
    main()
