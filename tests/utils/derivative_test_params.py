from typing import List, Tuple, Union

import numpy as np
from numpy import random as npr
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss

from tests.utils.official_catboost_objectives import OfficialLoglossObjective, OfficialMseObjective


class DerivativeTestParams:
    @classmethod
    def generate_test_params(cls) -> List[Tuple]:
        """
        :return: a list of test param tuples, each tuple containing:
                     preds: synthetic model predictions
                     targets: synthetic targets (y_true)
                     weights: synthetic sample weights or None (equal weight for all samples)
                     official_objective: an official CatBoost objective implementation, as a ground truth
                                         to compare our custom implementation to.
                     pytorch_loss_func: an official PyTorch loss function which is supposed to match the
                                        official CatBoost objective.

        """
        classification_params = [
            (preds, targets, weights, OfficialLoglossObjective(), BCEWithLogitsLoss(reduction="sum"))
            for preds, targets, weights in cls._generate_preds_targets_weights(targets_type="classification")
        ]
        regression_params = [
            (preds, targets, weights, OfficialMseObjective(), cls._mse_loss)
            for preds, targets, weights in cls._generate_preds_targets_weights(targets_type="regression")
        ]
        test_params = classification_params + regression_params
        return test_params

    @staticmethod
    def _generate_preds_targets_weights(
            targets_type: str,
            n_different: int = 10,
            vector_size: int = 100,
            random_seed: int = 34
    ) -> List[Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]]:
        assert targets_type in ("classification", "regression")

        random_state = npr.RandomState(random_seed)
        test_tuples = []
        for _ in range(n_different):
            preds = random_state.randn(vector_size)
            if targets_type == "classification":
                targets = random_state.randint(low=0, high=2, size=vector_size)
            elif targets_type == "regression":
                targets = random_state.randn(vector_size)
            weights = random_state.rand(vector_size)
            test_tuples.append((preds, targets, weights))
            test_tuples.append((preds, targets, None))
        return test_tuples

    @staticmethod
    def _mse_loss(preds: Tensor, targets: Tensor) -> Tensor:
        return 0.5 * MSELoss(reduction="sum")(preds, targets)
