# treeboost_autograd
## Easy Custom Losses for Tree Boosters using Pytorch
Why calculate first and second derivatives for your objective when you can let PyTorch do it for you?

This packages includes an easy to use custom PyTorch objective implementation for tree boosters (just add loss).\
**Supported boosting packages:** CatBoost, XGBoost, LightGBM.\
**Supported tasks:** regression, binary classification.

Check out the post in Towards Data Science: https://towardsdatascience.com/easy-custom-losses-for-tree-boosters-using-pytorch-57ffaa0b2eb3

## Usage
Usage is very similar for all boosting libraries:\
`from treeboost_autograd import CatboostObjective, LightGbmObjective, XgboostObjective`\
See `examples/example_classification.py` and `examples\example_regression.py` for ready-to-run examples.

`pip install treeboost_autograd`
```python
def absolute_error_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(preds - targets).sum()

custom_objective = CatboostObjective(loss_function=absolute_error_loss)
model = CatBoostRegressor(loss_function=custom_objective, eval_metric="MAE")
model.fit(X_train, y_train)
```
