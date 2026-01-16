import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random
import os
import subprocess
import atexit
import signal
import logging
from tu.utils.config import build_from_config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Union


logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed_benchmark(seed):
    # https://github.com/facebookresearch/deit/blob/ee8893c8063f6937fec7096e47ba324c206e22b9/main.py#L197
    logger.info(f'setting seed {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def spawn_ddp(args, worker):
    """

    Args:
        worker: a function with argument rank, world_size, args_in
            example see test_ddp_spawn

    Returns:

    """
    assert torch.cuda.is_available()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(worker, nprocs=world_size, args=(world_size, args), join=True)


def count_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('param count', param_count, 'model name', name)
    return param_count


def count_trainable_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    params = filter(lambda p: p.requires_grad, params)
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('trainable param count', param_count, 'model name', name)
    return param_count


def count_not_trainable_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    params = filter(lambda p: not p.requires_grad, params)
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('not trainable param count', param_count, 'model name', name)
    return param_count


def open_tensorboard(log_dir):
    # TODO automatically find another port if taken
    args = ["tensorboard", "--logdir", log_dir, '--bind_all', '--reload_multifile', 'True', '--load_fast', 'false']
    p = subprocess.Popen(args)

    def killme():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(killme)
    logger.info(" ".join(args))


def setup_ddp() -> float:
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    logger.info(f'setting up ddp: {dist.get_rank()} / {dist.get_world_size()}')
    assert dist.is_initialized()
    logger.info(f'use cuda device: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}')
    return rank


def setup_ddp_safe() -> float:
    if int(os.getenv('WORLD_SIZE', 0)) > 1:
        return setup_ddp()
    return 0


def build_dataloaders(cfg, seed, use_ddp) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # https://github.com/pytorch/elastic/blob/bc88e6982961d4117e53c4c8163ecf277f35c2c5/examples/imagenet/main.py#L268
    train_dataset = build_from_config(cfg['data'])
    
    # https://pytorch.org/docs/stable/data.html
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, seed=seed)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'],  # batch size per gpu
        # num_workers=2 if os.getenv('DEBUG') != '1' else 0,
        num_workers = 0, 
        shuffle=True if train_sampler is None else False, 
        pin_memory=True, drop_last=True,
        sampler=train_sampler,
    )
# train_dataset:blip_colors =['white']
    # blip_fruits =['clothes']
    # blip_mats =['season']
    # clip_features =tensor()
    # ground_truth_prompt_args =[['shirt'], ['spring'], ['red']]
    # gt_prompts =
    # ['a photo of a shirt for the spring season which is red in color']
    # images =tensor
    # inf_clip_features =tensor(
    # inf_color_prompts =['a photo of the color mytoken2']
    # inf_dict =
    # {'image': [tensor, 'prompt': ['a photo of a mytoken0 for the mytoken1 season which is mytoken2 in color'], 
    # 'gt_prompt': ['a photo of a gloves for the winter season which is red in color'], 
    # 'fruit_prompt': ['a photo of a mytoken0'], 
    # 'mat_prompt': ['a photo of the mytoken1 season'], 
    # 'color_prompt': ['a photo of the color mytoken2'], 
    # 'clip_feature': [tensor
    # inf_fruit_prompts =['a photo of a mytoken0']
    # inf_gt_prompts =['a photo of a gloves for the winter season which is red in color']
    # inf_images =tensor
    # inf_mat_prompts = ['a photo of the mytoken1 season']
    # inf_prompts = ['a photo of a mytoken0 for the mytoken1 season which is mytoken2 in color']
    # num_data_copies =1
    # num_data_per_prompt =1
    # num_placeholder_words =1
    # ph_words_all = ph_words_all = [['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken4', 'mytoken5', 'mytoken6'], ['mytoken4', 'mytoken5', 'mytoken6']](假如 num_data_per_prompt 为2)
    # [[array(['mytoken0'], dtype='<U8'), array(['mytoken1'], dtype='<U8'), array(['mytoken2'], dtype='<U8')]]
    # placeholder_words_prompt_args =
    # array([[['mytoken0']],       
            # [['mytoken1']],
            # [['mytoken2']]], dtype='<U8')
    # templates =['a photo of a {} for the {} season which is {} in color', 'a rendering of a {} for the {} season which is {} in color', 'a cropped photo of the {} for the {} season which is {} in color', 'the photo of a {} for the {} season which is {} in color', 'a photo of a clean {} for the {} season which is {} in color', 'a photo of a dirty {} for the {} season which is {} in color', 'a dark photo of the {} for the {} season which is {} in color', 'a photo of my {} for the {} season which is {} in color', 'a photo of the cool {} for the {} season which is {} in color', 'a close-up photo of a {} for the {} season which is {} in color', 'a bright photo of the {} for the {} season which is {} in color', 'a cropped photo of a {} for the {} season which is {} in color', 'a photo of the {} for the {} season which is {} in color', 'a good photo of the {} for the {} season which is {} in color', 'a photo of one {} for the {} season which is {} in color', 'a close-up photo of the {} for the {} season which is {} in color', 'a r...
    # unique_gt_words =[['shirt', 'spring', 'red']]
    # 
    logger.info(f'train dataset size: {len(train_dataset)}; numer of batches: {len(train_loader)}')
    val_dataset = build_from_config(cfg['data_val'], ref_dataset=train_dataset)
    # val_dataset = None
    if val_dataset is None:
        logger.info('val dataset is None')
        return train_loader, None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=min(cfg['training']['val_batch_size'], len(val_dataset)),
        # num_workers=2 if os.getenv('DEBUG') != '1' else 0, shuffle=False,
        num_workers=0, shuffle=False,

        pin_memory=True, drop_last=False,
    )
    logger.info(f'val dataset size: {len(val_dataset)}; numer of batches: {len(val_loader)}')
    return train_loader, val_loader


def build_training_modules(cfg, use_ddp=None, keys=None) -> Tuple[Dict[str, nn.Module], Dict[str, torch.optim.Optimizer], Dict[str, Union[torch.optim.lr_scheduler._LRScheduler, None]]]:
    modules = dict()
    optimizers = dict()
    schedulers = dict()

    if use_ddp:
        rank = dist.get_rank()

    if keys is None:
        keys = cfg['model'].keys()
    for key in keys:
        model = build_from_config(cfg['model'][key])  #* model = embeddings
        if use_ddp:
            model = model.cuda(rank)
            model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=True)
        else:
            model = model.cuda()
        optimizer = build_from_config(cfg['training']['optimizers'][key], params=model.parameters())
        scheduler = build_from_config(cfg['training']['schedulers'][key], optimizer=optimizer)
        modules[key] = model  #* modules[key] = embeddings
        optimizers[key] = optimizer
        schedulers[key] = scheduler

        logger.info(f"{key} params {count_parameters(modules[key], verbose=False)}")

    return modules, optimizers, schedulers
