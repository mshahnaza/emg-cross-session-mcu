import numpy as np
from .config import SEED, FT_LR, FT_EPOCHS
from .quantization import rnd, hidden_float, hidden_int8, dequant, predict_float, predict_int8


def prototypes(H, y, n_class):
    return np.stack([H[y == c].astype(np.float64).mean(axis=0) for c in range(n_class)]).astype(np.float32)


def predict_proto(H, P):
    d = ((H[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def cal_order(n):
    return np.random.default_rng(SEED).permutation(n)


def finetune(q, H, y, order, lr=FT_LR, epochs=FT_EPOCHS):
    lr = np.float32(lr)
    w = q['W'][2].copy()
    b = q['B'][2].copy()
    for _ in range(epochs):
        for n in order:
            h = H[n]
            z = w @ h + b
            e = np.exp(z - z.max()).astype(np.float32)
            p = e / e.sum()
            p[y[n]] -= np.float32(1)
            w -= np.outer(lr * p, h).astype(np.float32)
            b -= lr * p
    return w, b


def quantize_last(q, w, b):
    sw = np.float32(np.abs(w).max() / np.float32(127))
    qw = np.clip(rnd(w / sw), -127, 127).astype(np.int32)
    qb = rnd(b / (q['S_ACT'][2] * sw)).astype(np.int32)
    return qw, qb


def run_methods(q, Xn_train_stats, Xc_n, yc, Xt_n, yt):
    def pct(p):
        return float(np.mean(p == yt) * 100)

    n_class = q['W'][2].shape[0]
    res = {'No calibration': (pct(predict_float(q, Xn_train_stats)), pct(predict_int8(q, Xn_train_stats))),
           'Re-normalization': (pct(predict_float(q, Xt_n)), pct(predict_int8(q, Xt_n)))}
    Hf_c, Hf_t = hidden_float(q, Xc_n), hidden_float(q, Xt_n)
    Hq_c, Hq_t = hidden_int8(q, Xc_n), hidden_int8(q, Xt_n)
    res['Re-norm + prototypes'] = (
        pct(predict_proto(Hf_t, prototypes(Hf_c, yc, n_class))),
        pct(predict_proto(dequant(q, Hq_t), prototypes(dequant(q, Hq_c), yc, n_class))))
    order = cal_order(len(Xc_n))
    wf, bf = finetune(q, Hf_c, yc, order)
    wq, bq = finetune(q, dequant(q, Hq_c), yc, order)
    qw, qb = quantize_last(q, wq, bq)
    res['Re-norm + last-layer fine-tuning'] = (
        pct(np.argmax(Hf_t @ wf.T + bf, axis=1)),
        pct(np.argmax(Hq_t @ qw.T + qb, axis=1)))
    return res