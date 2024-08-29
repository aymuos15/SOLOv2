from functools import partial

from torch.utils.data import DataLoader

from .collate import collate
from .group_sampler import GroupSampler

def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     shuffle=True,
                     **kwargs):

    sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader

