from typing import Union, Callable

import numpy as np
import pytest
from torch import Tensor

from tests.utils.derivative_test_params import DerivativeTestParams
from tests.utils.official_catboost_objectives import OfficialLoglossObjective, OfficialMseObjective
from treeboost_autograd.booster_objectives import CatboostObjective


class TestCatboostObjective:
    @pytest.mark.parametrize("preds,targets,weights,official_objective,pytorch_loss_func",
                             DerivativeTestParams.generate_test_params())
    def test_derivative(self,
                        preds: np.ndarray,
                        targets: np.ndarray,
                        weights: Union[np.ndarray, None],
                        official_objective: Union[OfficialLoglossObjective, OfficialMseObjective],
                        pytorch_loss_func: Callable[[Tensor, Tensor], Tensor]):
        custom_pytorch_objective = CatboostObjective(pytorch_loss_func)
        result_pytorch = custom_pytorch_objective.calc_ders_range(preds, targets, weights)
        result_official = official_objective.calc_ders_range(preds, targets, weights)
        assert np.allclose(result_official, result_pytorch, rtol=1e-4)
