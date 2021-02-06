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
        preds = torch.FloatTensor(preds).requires_grad_()
        targets = torch.FloatTensor(targets)

        objective = - self.loss_function(preds, targets)
        der1, der2 = self._calculate_derivatives(objective, preds)

        if weights is not None:
            weights = np.asarray(weights)
            der1, der2 = weights * der1, weights * der2

        result = list(zip(der1, der2))
        return result

    @staticmethod
    def _calculate_derivatives(objective: Tensor, preds: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        der1, = torch.autograd.grad(objective, preds, create_graph=True)

        der2 = []
        for i in range(len(preds)):
            der2_i, = torch.autograd.grad(der1[i], preds, create_graph=True)
            der2_i = der2_i[i].item()
            der2.append(der2_i)

        der1 = der1.detach().numpy()
        der2 = np.array(der2)
        return der1, der2
