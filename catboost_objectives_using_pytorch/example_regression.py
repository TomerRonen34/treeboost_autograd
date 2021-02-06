import matplotlib.pyplot as plt
import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch import Tensor

from catboost_objectives_using_pytorch.custom_pytorch_objective import CustomPytorchObjective


def custom_regression_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """ penalize undershooting much more than overshooting """
    raw_diff = preds - targets
    is_undershoot = raw_diff < 0
    loss = torch.zeros_like(raw_diff)
    loss[is_undershoot] = torch.abs(raw_diff[is_undershoot] - 1) ** 3 - 1
    loss[~is_undershoot] = torch.abs(raw_diff[~is_undershoot])
    loss = loss.sum()
    return loss


def train_catboost_with_custom_loss():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)

    custom_objective = CustomPytorchObjective(loss_function=custom_regression_loss)
    model = CatBoostRegressor(loss_function=custom_objective, eval_metric="RMSE", n_estimators=300)

    model.fit(X_train, y_train)
    return model, X_test, y_test


def plot_relative_difference(model, X_test, y_test):
    pred_test = model.predict(X_test)
    rel_diff = (pred_test - y_test) / y_test

    plt.hist(rel_diff, bins=np.arange(-0.5, 1.51, 0.1))
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel("relative difference = (preds - targets) / targets")
    plt.show()


def main():
    model, X_test, y_test = train_catboost_with_custom_loss()
    plot_relative_difference(model, X_test, y_test)


if __name__ == '__main__':
    main()
