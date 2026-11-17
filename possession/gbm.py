"""Dependency-free inference for the gradient-boosted trees (numpy only).

The model is trained with scikit-learn (training/train.py) and exported to a flat .npz, so that
producing the submission needs nothing but numpy and gives the same numbers on every machine."""
import numpy as np


def export(clf, names, path, extra=None):
    """Flatten a fitted sklearn HistGradientBoostingClassifier into arrays."""
    feat, thr, left, right, miss_left, value, leaf, offsets = [], [], [], [], [], [], [], [0]
    n_iter = len(clf._predictors)
    n_class = len(clf._predictors[0])
    for it in clf._predictors:
        for tree in it:
            nd = tree.nodes
            assert not nd["is_categorical"].any()
            feat.append(nd["feature_idx"].astype(np.int32))
            thr.append(nd["num_threshold"].astype(np.float64))
            left.append(nd["left"].astype(np.int32))
            right.append(nd["right"].astype(np.int32))
            miss_left.append(nd["missing_go_to_left"].astype(bool))
            value.append(nd["value"].astype(np.float64))
            leaf.append(nd["is_leaf"].astype(bool))
            offsets.append(offsets[-1] + len(nd))
    np.savez_compressed(
        path, feat=np.concatenate(feat), thr=np.concatenate(thr), left=np.concatenate(left),
        right=np.concatenate(right), miss_left=np.concatenate(miss_left), value=np.concatenate(value),
        leaf=np.concatenate(leaf), offsets=np.array(offsets), n_iter=n_iter, n_class=n_class,
        baseline=np.asarray(clf._baseline_prediction, np.float64).ravel(), names=np.array(names),
        **(extra or {}))


class Model:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        self.names = [str(n) for n in self.z["names"]]
        self.n_iter = int(self.z["n_iter"])
        self.n_class = int(self.z["n_class"])

    def predict_proba(self, X, chunk=40000):
        z = self.z
        X = np.asarray(X, np.float64)
        out = np.empty((len(X), self.n_class))
        for c0 in range(0, len(X), chunk):
            Xc = X[c0:c0 + chunk]
            raw = np.tile(z["baseline"], (len(Xc), 1))
            rows = np.arange(len(Xc))
            t = 0
            for _ in range(self.n_iter):
                for k in range(self.n_class):
                    a = z["offsets"][t]
                    t += 1
                    node = np.zeros(len(Xc), np.int32) + a
                    active = ~z["leaf"][node]
                    while active.any():
                        n = node[active]
                        x = Xc[rows[active], z["feat"][n]]
                        go_left = np.where(np.isnan(x), z["miss_left"][n], x <= z["thr"][n])
                        node[active] = a + np.where(go_left, z["left"][n], z["right"][n])
                        active = ~z["leaf"][node]
                    raw[:, k] += z["value"][node]
            raw -= raw.max(axis=1, keepdims=True)
            e = np.exp(raw)
            out[c0:c0 + chunk] = e / e.sum(axis=1, keepdims=True)
        return out
