from typing import Callable, Tuple

import numpy as np
from torch import Tensor

from treeboost_autograd.pytorch_objective import PytorchObjective


class LightGbmObjective:
    def __init__(self, loss_function: Callable[[Tensor, Tensor], Tensor]):
        self.pytorch_objective = PytorchObjective(loss_function, use_minus_loss_as_objective=False)

    def __call__(self,
                 targets: np.ndarray,
                 preds: np.ndarray,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        deriv1, deriv2 = self.pytorch_objective.calculate_derivatives(preds, targets)
        return deriv1, deriv2
