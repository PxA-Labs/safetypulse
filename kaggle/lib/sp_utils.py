"""SafetyPulse helpers shared by the Kaggle notebooks.

Upload this file to Kaggle once (as a private dataset, e.g. `safetypulse-lib`, or as a
Utility Script) and attach it to every notebook. The first cell of each notebook finds it.

Plain PyTorch + transformers only. No Lexsi toolkit is imported here: toolkit calls live in
the notebooks' toolkit cells until their real APIs are confirmed (docs/03 §5, ADR-0003).

Residual-stream index convention used everywhere:
    resid index 0      = token embeddings
    resid index i >= 1 = output of decoder layer i-1
so a model with L layers has L+1 resid positions.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

