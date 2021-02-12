import argparse
from typing import Union, Callable

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import SoftMarginLoss

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from treeboost_autograd.catboost_objective import CatboostObjective
from treeboost_autograd.xgboost_objective import XgboostObjective
from treeboost_autograd.lightgbm_objective import LightGbmObjective

VALID_BOOSTING_PACKAGES = ["catboost", "xgboost", "lightgbm"]


def main():
    args = _parse_args()
    train_and_eval_custom_classifier(args.boosting_package)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boosting_package', type=str,
                        choices=VALID_BOOSTING_PACKAGES,
                        default=VALID_BOOSTING_PACKAGES[0])

    args = parser.parse_args()
    return args


def train_and_eval_custom_classifier(boosting_package: str,
                                     n_estimators: int = 10, verbose: bool = True, random_seed: int = 2021):
    assert boosting_package in VALID_BOOSTING_PACKAGES
    if verbose:
        _print_title(boosting_package)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

    custom_loss_function = _soft_margin_loss
    model = _fit_custom_classifier(boosting_package,
                                   X_train, y_train, custom_loss_function,
                                   n_estimators, verbose, random_seed)
    pred_test = model.predict(X_test)
    macro_f1_score = f1_score(y_test, pred_test, average="macro")

    if verbose:
        print(classification_report(y_test, pred_test))

    return macro_f1_score


def _soft_margin_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """ transform binary targets from {0,1} to {-1,1}, then use the standard SoftMarginLoss """
    targets = 2 * targets - 1
    return SoftMarginLoss(reduction="sum")(preds, targets)


def _fit_custom_classifier(boosting_package: str,
                           X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                           n_estimators: int, verbose: bool, random_seed: int
                           ) -> Union[CatBoostClassifier, XGBClassifier, LGBMClassifier]:
    if boosting_package == "catboost":
        return _fit_custom_catboost_classifier(X_train, y_train, custom_loss_function,
                                               n_estimators, verbose, random_seed)
    elif boosting_package == "xgboost":
        return _fit_custom_xgboost_classifier(X_train, y_train, custom_loss_function,
                                              n_estimators, verbose, random_seed)
    elif boosting_package == "lightgbm":
        return _fit_custom_lightgbm_classifier(X_train, y_train, custom_loss_function,
                                               n_estimators, verbose, random_seed)
    else:
        raise ValueError(f"Unsupported boosting_package '{boosting_package}'")


def _fit_custom_catboost_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                    n_estimators: int, verbose: bool, random_seed: int
                                    ) -> CatBoostClassifier:
    custom_objective = CatboostObjective(loss_function=custom_loss_function)
    model = CatBoostClassifier(loss_function=custom_objective, n_estimators=n_estimators, random_seed=random_seed,
                               eval_metric="ZeroOneLoss", allow_writing_files=False)
    model.fit(X_train, y_train, verbose=verbose)
    return model


def _fit_custom_xgboost_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                   n_estimators: int, verbose: bool, random_seed: int
                                   ) -> XGBClassifier:
    custom_objective = XgboostObjective(loss_function=custom_loss_function)
    model = XGBClassifier(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="error", eval_set=[(X_train, y_train)], verbose=verbose)
    return model


def _fit_custom_lightgbm_classifier(X_train: np.ndarray, y_train: np.ndarray, custom_loss_function: Callable,
                                    n_estimators: int, verbose: bool, random_seed: int
                                    ) -> LGBMClassifier:
    custom_objective = LightGbmObjective(loss_function=custom_loss_function)
    model = LGBMClassifier(objective=custom_objective, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train, eval_metric="binary_error", eval_set=(X_train, y_train), verbose=verbose)
    return model


def _print_title(boosting_package: str) -> None:
    title = f"Training and evaluating custom '{boosting_package}' classifier"
    underline = '=' * len(title)
    print(title + '\n' + underline)


if __name__ == '__main__':
    main()
