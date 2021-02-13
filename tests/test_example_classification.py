from itertools import product
from typing import Callable

import pytest

from examples.example_classification import train_and_eval_custom_classifier, soft_margin_loss, squared_hinge_loss


class TestTrainAndEvalCustomClassifier:
    @pytest.mark.parametrize("boosting_package,custom_loss_function",
                             product(["catboost", "xgboost", "lightgbm"], [soft_margin_loss, squared_hinge_loss]))
    def test_train_and_eval_custom_classifier(self, boosting_package: str, custom_loss_function: Callable):
        macro_f1_score = train_and_eval_custom_classifier(boosting_package, custom_loss_function)
        assert macro_f1_score > 0.9
