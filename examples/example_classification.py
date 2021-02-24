from typing import Union, Callable

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import SoftMarginLoss

from treeboost_autograd import CatboostObjective, LightGbmObjective, XgboostObjective


def main():
    train_and_eval_custom_classifier(boosting_package="catboost", custom_loss_function=squared_hinge_loss)
    train_and_eval_custom_classifier(boosting_package="xgboost", custom_loss_function=squared_hinge_loss)
    train_and_eval_custom_classifier(boosting_package="lightgbm", custom_loss_function=squared_hinge_loss)


def soft_margin_loss(preds: Tensor, targets: Tensor) -> Tensor:
    targets = 2 * targets - 1
    loss = SoftMarginLoss(reduction="sum")(preds, targets)
    return loss


def squared_hinge_loss(preds: Tensor, targets: Tensor) -> Tensor:
    targets = 2 * targets - 1
    loss = torch.max(Tensor([0]), 1 - preds * targets) ** 2
    loss = loss.sum()
    return loss


def train_and_eval_custom_classifier(boosting_package: str, custom_loss_function: Callable[[Tensor, Tensor], Tensor],
                                     n_estimators: int = 10, random_seed: int = 2021) -> float:
    try:
        assert boosting_package in ["catboost", "xgboost", "lightgbm"]
        _print_title(boosting_package)

        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

        model = _fit_custom_classifier(boosting_package,
                                       X_train, y_train, custom_loss_function,
                                       n_estimators, random_seed)

        pred_test = model.predict(X_test)
        macro_f1_score = f1_score(y_test, pred_test, average="macro")

        print(classification_report(y_test, pred_test))
        return macro_f1_score

    except ImportError:
        print(f"Woops! We can't run the example for '{boosting_package}', probably because it isn't installed")


def _fit_custom_classifier(boosting_package: str, *args, **kwargs
                           ) -> Union["CatBoostClassifier", "XGBClassifier", "LGBMClassifier"]:
    boosting_package_2_fit_function = {"catboost": _fit_custom_catboost_classifier,
                                       "xgboost": _fit_custom_xgboost_classifier,
                                       "lightgbm": _fit_custom_lightgbm_classifier}
    fit_function = boosting_package_2_fit_function[boosting_package]
    return fit_function(*args, **kwargs)


def _fit_custom_catboost_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                    n_estimators: int, random_seed: int
                                    ) -> "CatBoostClassifier":
    from catboost import CatBoostClassifier
    custom_objective = CatboostObjective(loss_function=custom_loss_function)
    model = CatBoostClassifier(loss_function=custom_objective, n_estimators=n_estimators, random_seed=random_seed,
                               eval_metric="ZeroOneLoss", allow_writing_files=False)
    model.fit(X_train, y_train, verbose=True)
    return model


def _fit_custom_xgboost_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                   n_estimators: int, random_seed: int
                                   ) -> "XGBClassifier":
    from xgboost import XGBClassifier
    custom_objective = XgboostObjective(loss_function=custom_loss_function)
    model = XGBClassifier(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="error", eval_set=[(X_train, y_train)], verbose=True)
    return model


def _fit_custom_lightgbm_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                    n_estimators: int, random_seed: int
                                    ) -> "LGBMClassifier":
    from lightgbm import LGBMClassifier
    custom_objective = LightGbmObjective(loss_function=custom_loss_function)
    model = LGBMClassifier(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="binary_error", eval_set=(X_train, y_train), verbose=True)
    return model


def _print_title(boosting_package: str) -> None:
    title = f"Training and evaluating custom '{boosting_package}' classifier"
    underline = '=' * len(title)
    print('\n\n' + title + '\n' + underline)


if __name__ == '__main__':
    main()
