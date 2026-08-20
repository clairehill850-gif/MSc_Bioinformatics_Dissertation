#!/usr/bin/env python3
# What this does: this script has the shared deterministic-inference settings, so replicated runs agree.
import os

import torch


def enable():
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        print("warning: CUBLAS_WORKSPACE_CONFIG unset - "
              "run 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' before python")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class null_autocast:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
