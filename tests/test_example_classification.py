import pytest
from examples.example_classification import VALID_BOOSTING_PACKAGES
from examples.example_classification import train_and_eval_custom_classifier


class TestTrainAndEvalCustomClassifier:
    @pytest.mark.parametrize("boosting_package", VALID_BOOSTING_PACKAGES)
    def test_train_and_eval_custom_classifier(self, boosting_package: str):
        macro_f1_score = train_and_eval_custom_classifier(boosting_package, verbose=False)
        assert macro_f1_score > 0.9
