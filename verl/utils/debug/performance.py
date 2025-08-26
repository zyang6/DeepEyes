# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.distributed as dist

from verl.utils.logger.aggregate_logger import DecoratorLoggerBase
from verl.utils.device import get_torch_device


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        message = f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}"

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


class GPUMemoryLogger(DecoratorLoggerBase):
    """_summary_
    
    Usage:
        For example, in actor function, we initialize a GPUMemoryLogger
        
        ```
        from verl.utils.debug.performance import GPUMemoryLogger
        @GPUMemoryLogger(role="actor")
        def update_actor(self, batch):
            # do something
            return
        ```
    
    """
    
    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)
    
    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)
        return f
    
    def log(self, func, *args, **kwargs):
        memory_allocated = get_torch_device().memory_allocated() / 1024**3
        memory_reserved = get_torch_device().memory_reserved() / 1024**3

        message = f"Before {func.__name__}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}"
        self.logging_function(message)
        output = func(*args, **kwargs)
        message = f"After {func.__name__}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}"
        self.logging_function(message)
        return output