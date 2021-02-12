from typing import Callable, Sequence, List, Tuple

from torch import Tensor

from treeboost_autograd.pytorch_objective import PytorchObjective


class CatboostObjective:
    def __init__(self, loss_function: Callable[[Tensor, Tensor], Tensor]):
        self.pytorch_objective = PytorchObjective(loss_function, use_minus_loss_as_objective=True)

    def calc_ders_range(self,
                        preds: Sequence[float],
                        targets: Sequence[float],
                        weights: Sequence[float] = None
                        ) -> List[Tuple[float, float]]:
        """  Calculates first and second derivatives of the objective (-loss) w.r.t preds.  """
        deriv1, deriv2 = self.pytorch_objective.calculate_derivatives(preds, targets, weights)
        result = list(zip(deriv1, deriv2))
        return result
