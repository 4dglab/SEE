import os

import numpy as np
import pynvml
import torch
import torch.distributed as dist

from scce.utils import LOGGER


def find_devices(world_size: int = 1):
    if world_size <= 0:
        raise Exception("world_size <= 0")
    try:
        pynvml.nvmlInit()
        LOGGER.info("Found %d GPU(s)" % pynvml.nvmlDeviceGetCount())

        free_mems = np.array(
            [
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        )
        if not free_mems.size:
            raise Exception("No GPU available.")

        used_devices = np.argpartition(free_mems, -world_size)[-world_size:]
        used_devices = used_devices[free_mems[used_devices] > 0]
        used_devices = used_devices[np.argsort(free_mems[used_devices])][::-1]
    except pynvml.NVMLError:
        raise Exception("No GPU available.")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(used_devices.astype(str))
    LOGGER.info("Using GPU %d as computation device.", used_devices)


def init_dist(rank, world_size) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt
