import numpy as np
import numpy.random as npr
import pytest
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from typing import List, Tuple, Union, Callable

from tests.official_catboost_objectives import OfficialLoglossObjective, OfficialMseObjective
from treeboost_autograd.booster_objectives import CatboostObjective


class _TestParams:
    @classmethod
    def generate_test_params(cls) -> List[Tuple]:
        classification_params = [
            (approxes, targets, weights, OfficialLoglossObjective(), BCEWithLogitsLoss(reduction="sum"))
            for approxes, targets, weights in cls._generate_approxes_targets_weights(targets_type="classification")
        ]
        regression_params = [
            (approxes, targets, weights, OfficialMseObjective(), cls._mse_loss)
            for approxes, targets, weights in cls._generate_approxes_targets_weights(targets_type="regression")
        ]
        test_params = classification_params + regression_params
        return test_params

    @staticmethod
    def _generate_approxes_targets_weights(
            targets_type: str,
            n_different: int = 10,
            vector_size: int = 100,
            random_seed: int = 34
    ) -> List[Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]]:
        assert targets_type in ("classification", "regression")

        random_state = npr.RandomState(random_seed)
        test_tuples = []
        for _ in range(n_different):
            approxes = random_state.randn(vector_size)
            if targets_type == "classification":
                targets = random_state.randint(low=0, high=2, size=vector_size)
            elif targets_type == "regression":
                targets = random_state.randn(vector_size)
            weights = random_state.rand(vector_size)
            test_tuples.append((approxes, targets, weights))
            test_tuples.append((approxes, targets, None))
        return test_tuples

    @staticmethod
    def _mse_loss(approxes: Tensor, targets: Tensor) -> Tensor:
        return 0.5 * MSELoss(reduction="sum")(approxes, targets)


class TestCatboostObjective:
    @pytest.mark.parametrize("approxes,targets,weights,official_objective,pytorch_loss_func",
                             _TestParams.generate_test_params())
    def test_calc_ders_range(self,
                             approxes: np.ndarray,
                             targets: np.ndarray,
                             weights: Union[np.ndarray, None],
                             official_objective: Union[OfficialLoglossObjective, OfficialMseObjective],
                             pytorch_loss_func: Callable[[Tensor, Tensor], Tensor]):
        custom_pytorch_objective = CatboostObjective(pytorch_loss_func)
        result_pytorch = custom_pytorch_objective.calc_ders_range(approxes, targets, weights)
        result_official = official_objective.calc_ders_range(approxes, targets, weights)
        assert np.allclose(result_official, result_pytorch, rtol=1e-4)
