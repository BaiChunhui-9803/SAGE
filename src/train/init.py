import torch
import torch.nn as nn
import numpy as np
import random  # 引入随机库
import logging

def set_seed(seed=42):
    """
    固定全局随机种子，确保实验可重复性（确定性）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保卷积等底层操作也是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"全局随机种子已设置为: {seed}")
