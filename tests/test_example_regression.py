import pytest

from examples.example_regression import train_and_eval_custom_regressor, dont_undershoot_loss


class TestTrainAndEvalCustomClassifier:
    @pytest.mark.parametrize("boosting_package,n_estimators",
                             [("xgboost", 100), ("lightgbm", 100), ("catboost", 300)])
    def test_train_and_eval_custom_classifier(self, boosting_package: str, n_estimators: int):
        mean_abs_rel_diff, undershoot_percentage = train_and_eval_custom_regressor(
            boosting_package, dont_undershoot_loss, n_estimators, is_plot=False)
        assert mean_abs_rel_diff < 0.3
        assert undershoot_percentage < 0.3
