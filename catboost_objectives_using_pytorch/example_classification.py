from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import SoftMarginLoss

from catboost_objectives_using_pytorch.custom_pytorch_objective import CustomPytorchObjective


def soft_margin_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """ transform binary targets from {0,1} to {-1,1}, then use the standard SoftMarginLoss """
    targets = 2 * targets - 1
    return SoftMarginLoss()(preds, targets)


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)

    custom_objective = CustomPytorchObjective(loss_function=soft_margin_loss)
    model = CatBoostClassifier(loss_function=custom_objective, n_estimators=100, eval_metric="Accuracy")

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    print(classification_report(y_test, pred_test))


if __name__ == '__main__':
    main()
