
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__)) # 获取当前绝对路径C
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]				 # 上一级目录B
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from torch.utils import data
from dataset.data_reader import CTDataset,MRDataset,CTAbdomenDataset,MRAbdomenDataset
from model.deeplabv2 import get_deeplab_v2
from domain_adaptation.config import cfg, cfg_from_file
from domain_adaptation.train_UDA_ablation_new import train_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default='',
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    return parser.parse_args()


def main():
    #LOAD ARGS
    args = get_arguments()
    print('Called with args')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    #auto-generate exp name if not specified

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}/CVPR_{cfg.SOURCE}2{cfg.TARGET}'
    pth = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # auto-generate snapshot path if not specified

    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # Initialization
    _init_fn = None

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)
    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    assert osp.exists(cfg.TRAIN.EMA_RESTORE_FROM), f'Missing init model {cfg.TRAIN.EMA_RESTORE_FROM}'

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        ema_model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL, ema=True)

        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)


        saved_ema_state_dict = torch.load(cfg.TRAIN.EMA_RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.EMA_RESTORE_FROM:
            new_params = ema_model.state_dict().copy()
            for i in saved_ema_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_ema_state_dict[i]
            ema_model.load_state_dict(new_params)
        else:
            ema_model.load_state_dict(saved_ema_state_dict)

        print('Model and EMA Model loaded !')

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    # DataLoaders
    train_mr_pth = '/home/data_backup/abdominalDATA/train_mr.txt'
    train_ct_pth = '/home/data_backup/abdominalDATA/train_ct.txt'

    val_mr_pth   = '/home/data_backup/abdominalDATA/val_mr.txt'
    val_ct_pth   = '/home/data_backup/abdominalDATA/val_ct.txt'

    transforms = None
    img_mean = cfg.TRAIN.IMG_MEAN
    mrtrain_dataset = MRAbdomenDataset(data_pth=train_mr_pth, 
                                img_mean=img_mean, transform=transforms)

    cttrain_dataset = CTAbdomenDataset(data_pth=train_ct_pth,
                                img_mean=img_mean, transform=transforms)

    mrval_dataset   = MRAbdomenDataset(data_pth=val_mr_pth, 
                              img_mean=img_mean, transform=transforms)

    ctval_dataset   = CTAbdomenDataset(data_pth=val_ct_pth,  img_mean=img_mean,
                              transform=transforms)

    if cfg.SOURCE == 'MR':
        strain_dataset = mrtrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = cttrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)
        sval_dataset = mrval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)

        trgval_dataset = ctval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)

    elif cfg.SOURCE == 'CT':

        strain_dataset = cttrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = mrtrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)
        sval_dataset = ctval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)

        trgval_dataset = mrval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)

    print('dataloader finish')
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(model, ema_model, strain_loader, trgtrain_loader, sval_loader, cfg)


if __name__ == '__main__':
    main()
