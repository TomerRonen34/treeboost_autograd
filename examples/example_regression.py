from typing import Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch import Tensor

from treeboost_autograd import CatboostObjective, LightGbmObjective, XgboostObjective


def main():
    _plot_custom_loss(dont_undershoot_loss)
    train_and_eval_custom_regressor("xgboost", dont_undershoot_loss, n_estimators=100)
    train_and_eval_custom_regressor("lightgbm", dont_undershoot_loss, n_estimators=100)
    train_and_eval_custom_regressor("catboost", dont_undershoot_loss, n_estimators=300)


def dont_undershoot_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """ penalize undershooting much more than overshooting (regression loss) """
    raw_diff = preds - targets
    is_undershoot = raw_diff < 0
    loss = torch.zeros_like(raw_diff)
    loss[is_undershoot] = torch.abs(raw_diff[is_undershoot] - 1) ** 4 - 1
    loss[~is_undershoot] = torch.abs(raw_diff[~is_undershoot])
    loss = loss.sum()
    return loss


def train_and_eval_custom_regressor(boosting_package: str, custom_loss_function: Callable[[Tensor, Tensor], Tensor],
                                    n_estimators: int, random_seed: int = 2021, is_plot: bool = True
                                    ) -> Tuple[float, float]:
    try:
        assert boosting_package in ["catboost", "xgboost", "lightgbm"]
        _print_title(boosting_package)

        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)

        model = _fit_custom_regressor(boosting_package,
                                      X_train, y_train, custom_loss_function,
                                      n_estimators, random_seed)

        pred_test = model.predict(X_test)
        mean_abs_rel_diff = _calc_mean_absolute_relative_difference(pred_test, y_test)
        undershoot_percentage = _calc_undershoot_percentage(pred_test, y_test)

        if is_plot:
            plot_title = _generate_plot_title(boosting_package, mean_abs_rel_diff, undershoot_percentage)
            _plot_relative_difference(pred_test, y_test, title=plot_title)

        return mean_abs_rel_diff, undershoot_percentage

    except ImportError:
        print(f"Woops! We can't run the example for '{boosting_package}', probably because it isn't installed")


def _fit_custom_regressor(boosting_package: str, *args, **kwargs
                          ) -> Union["CatBoostRegressor", "XGBRegressor", "LGBMRegressor"]:
    boosting_package_2_fit_function = {"catboost": _fit_custom_catboost_regressor,
                                       "xgboost": _fit_custom_xgboost_regressor,
                                       "lightgbm": _fit_custom_lightgbm_regressor}
    fit_function = boosting_package_2_fit_function[boosting_package]
    return fit_function(*args, **kwargs)


def _fit_custom_catboost_regressor(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                   n_estimators: int, random_seed: int
                                   ) -> "CatBoostRegressor":
    from catboost import CatBoostRegressor
    custom_objective = CatboostObjective(loss_function=custom_loss_function)
    model = CatBoostRegressor(loss_function=custom_objective, n_estimators=n_estimators, random_seed=random_seed,
                              eval_metric="MAE", allow_writing_files=False)
    model.fit(X_train, y_train, verbose=True)
    return model


def _fit_custom_xgboost_regressor(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                  n_estimators: int, random_seed: int
                                  ) -> "XGBRegressor":
    from xgboost import XGBRegressor
    custom_objective = XgboostObjective(loss_function=custom_loss_function)
    model = XGBRegressor(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_train, y_train)], verbose=True)
    return model


def _fit_custom_lightgbm_regressor(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                   n_estimators: int, random_seed: int
                                   ) -> "LGBMRegressor":
    from lightgbm import LGBMRegressor
    custom_objective = LightGbmObjective(loss_function=custom_loss_function)
    model = LGBMRegressor(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="mae", eval_set=(X_train, y_train), verbose=True)
    return model


def _calc_relative_difference(pred_test: np.ndarray, y_test: np.ndarray) -> float:
    rel_diff = (pred_test - y_test) / y_test
    return rel_diff


def _calc_mean_absolute_relative_difference(pred_test: np.ndarray, y_test: np.ndarray) -> float:
    rel_diff = _calc_relative_difference(pred_test, y_test)
    mean_abs_rel_diff = float(np.mean(np.abs(rel_diff)))
    return mean_abs_rel_diff


def _calc_undershoot_percentage(pred_test: np.ndarray, y_test: np.ndarray) -> float:
    rel_diff = _calc_relative_difference(pred_test, y_test)
    undershoot_percentage = float(np.mean(rel_diff < 0))
    return undershoot_percentage


def _plot_relative_difference(pred_test: np.ndarray, y_test: np.ndarray, title: str) -> None:
    rel_diff = _calc_relative_difference(pred_test, y_test)

    plt.hist(rel_diff, bins=np.arange(-1, 1.51, 0.1))
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel("relative difference = (preds - targets) / targets")
    plt.title(title)
    plt.show()


def _print_title(boosting_package: str) -> None:
    title = f"Training and evaluating custom '{boosting_package}' regressor"
    underline = '=' * len(title)
    print('\n\n' + title + '\n' + underline)


def _generate_plot_title(boosting_package: str, mean_abs_rel_diff: float, undershoot_percentage: float) -> str:
    plot_title = f"{boosting_package} regressor\n" \
                 f"mean absolute relative difference = {mean_abs_rel_diff:.2f}\n" \
                 f"undershoot percentage = {undershoot_percentage:.2f}"
    return plot_title


def _plot_custom_loss(custom_loss_function: Callable) -> None:
    preds = np.linspace(-0.5, 0.5, 101)
    targets = np.zeros_like(preds)
    with torch.no_grad():
        loss = np.hstack([custom_loss_function(Tensor([p]), Tensor([t])).numpy()
                          for p, t in zip(preds, targets)])
    plt.plot(preds, loss)
    plt.title("Custom regression loss function - don't undershoot!")
    plt.xlabel("diff = (preds - targets)")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    main()
