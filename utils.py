# utils.py
# 通用工具函数（与LCNN项目共用或类似）

import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml


def init_weights(module):
    """
    初始化模型权重
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def read_yaml(config_path):
    """
    读取YAML配置文件
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed: int):
    """
    设置所有随机种子以保证可复现性
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False