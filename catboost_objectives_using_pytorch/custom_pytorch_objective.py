from typing import Callable, Sequence, List, Tuple

import numpy as np
import torch
from torch import Tensor


class CustomPytorchObjective:
    def __init__(self, loss_function: Callable[[Tensor, Tensor], Tensor]):
        self.loss_function = loss_function

    def calc_ders_range(self,
                        preds: Sequence[float],
                        targets: Sequence[float],
                        weights: Sequence[float] = None
                        ) -> List[Tuple[float, float]]:
        """  Calculates first and second derivatives of the objective (-loss) w.r.t preds.  """
        preds = torch.FloatTensor(preds).requires_grad_()
        targets = torch.FloatTensor(targets)

        objective = - self.loss_function(preds, targets)
        deriv1, deriv2 = self._calculate_derivatives(objective, preds)

        if weights is not None:
            weights = np.asarray(weights)
            deriv1, deriv2 = weights * deriv1, weights * deriv2

        result = list(zip(deriv1, deriv2))
        return result

    @staticmethod
    def _calculate_derivatives(objective: Tensor, preds: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        deriv1, = torch.autograd.grad(objective, preds, create_graph=True)

        deriv2 = []
        for i in range(len(preds)):
            deriv2_i, = torch.autograd.grad(deriv1[i], preds, retain_graph=True)
            deriv2_i = deriv2_i[i].item()
            deriv2.append(deriv2_i)

        deriv1 = deriv1.detach().numpy()
        deriv2 = np.array(deriv2)
        return deriv1, deriv2
