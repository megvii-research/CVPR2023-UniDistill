# flake8: noqa W605
import importlib
import os
import re
import subprocess
import sys
import warnings
from collections import defaultdict

import numpy as np
import PIL
import torch
import torchvision
from tabulate import tabulate

__all__ = ["collect_env_info"]


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
