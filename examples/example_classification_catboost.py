from typing import Callable

from catboost import CatBoostClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.nn import SoftMarginLoss

from treeboost_autograd.booster_objectives import CatboostObjective


def soft_margin_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """ transform binary targets from {0,1} to {-1,1}, then use the standard SoftMarginLoss """
    targets = 2 * targets - 1
    loss = SoftMarginLoss(reduction="sum")(preds, targets)
    return loss


def squared_hinge_loss(preds: Tensor, targets: Tensor) -> Tensor:
    targets = 2 * targets - 1
    loss = torch.max(Tensor([0]), 1 - preds * targets) ** 2
    loss = loss.sum()
    return loss


def train_and_eval_custom_classifier(custom_loss_function: Callable[[Tensor, Tensor], Tensor] = soft_margin_loss,
                                     n_estimators: int = 10, verbose: bool = True, random_seed: int = 2021) -> float:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

    custom_objective = CatboostObjective(loss_function=custom_loss_function)
    model = CatBoostClassifier(loss_function=custom_objective, n_estimators=n_estimators, random_seed=random_seed,
                               eval_metric="ZeroOneLoss", allow_writing_files=False)
    model.fit(X_train, y_train, verbose=verbose)

    pred_test = model.predict(X_test)
    macro_f1_score = f1_score(y_test, pred_test, average="macro")

    print(classification_report(y_test, pred_test))
    return macro_f1_score


if __name__ == '__main__':
    train_and_eval_custom_classifier(custom_loss_function=squared_hinge_loss)
    train_and_eval_custom_classifier(custom_loss_function=soft_margin_loss)
