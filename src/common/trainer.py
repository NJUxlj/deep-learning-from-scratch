# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络的训练的类
    """