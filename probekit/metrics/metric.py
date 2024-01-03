from typing import List, Any, Dict, Optional
import torch

from probekit.utils.types import PyTorchDevice
from probekit.utils.dataset import ClassificationDataset
from probekit.models.probe import Probe


class Metric:
    def __init__(self):
        pass

    def compute(self, *args, **kwargs) -> Any:
        return self._compute(*args, **kwargs)

    def _compute(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
