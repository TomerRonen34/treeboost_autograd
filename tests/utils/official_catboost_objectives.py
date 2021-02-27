import numpy as np


class OfficialLoglossObjective:
    """ An official Catboost example """

    def calc_ders_range(self, preds, targets, weights=None):
        assert len(preds) == len(targets)
        if weights is not None:
            assert len(weights) == len(preds)

        result = []
        for index in range(len(targets)):
            e = np.exp(preds[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class OfficialMseObjective:
    """ An official Catboost example """

    def calc_ders_range(self, preds, targets, weights):
        assert len(preds) == len(targets)
        if weights is not None:
            assert len(weights) == len(preds)

        result = []
        for index in range(len(targets)):
            der1 = targets[index] - preds[index]
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result
