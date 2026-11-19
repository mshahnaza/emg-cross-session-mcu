import numpy as np
from .config import SEED


def rnd(v):
    return np.floor(v + 0.5)


def act_scale(a):
    return float(np.percentile(np.abs(a), 99.9) / 127.0) + 1e-12


def quantize_mlp(mlp, X_calib, n=5000):
    W = [w.T.astype(np.float32) for w in mlp.coefs_]
    B = [b.astype(np.float32) for b in mlp.intercepts_]
    idx = np.random.default_rng(SEED).choice(len(X_calib), min(n, len(X_calib)), replace=False)
    h = X_calib[idx]
    H = [h]
    for i in range(2):
        h = np.maximum(h @ W[i].T + B[i], 0)
        H.append(h)
    s_act = np.array([act_scale(a) for a in H], dtype=np.float32)
    s_w = np.array([np.abs(w).max() / 127.0 for w in W], dtype=np.float32)
    s_mul = (s_act * s_w).astype(np.float32)
    qw = [np.clip(rnd(W[i] / s_w[i]), -127, 127).astype(np.int8) for i in range(3)]
    qb = [rnd(B[i] / s_mul[i]).astype(np.int32) for i in range(3)]
    return {'W': W, 'B': B, 'S_ACT': s_act, 'S_MUL': s_mul, 'QW': qw, 'QB': qb}


def hidden_float(q, Xn):
    h = np.maximum(Xn @ q['W'][0].T + q['B'][0], 0)
    return np.maximum(h @ q['W'][1].T + q['B'][1], 0).astype(np.float32)


def hidden_int8(q, Xn):
    h = np.clip(rnd(Xn / q['S_ACT'][0]), -127, 127).astype(np.int32)
    for i in range(2):
        a = h @ q['QW'][i].astype(np.int32).T + q['QB'][i]
        real = np.maximum(a.astype(np.float32) * q['S_MUL'][i], np.float32(0))
        h = np.clip(rnd(real / q['S_ACT'][i + 1]), -127, 127).astype(np.int32)
    return h


def dequant(q, h):
    return h.astype(np.float32) * q['S_ACT'][2]


def predict_float(q, Xn):
    return np.argmax(hidden_float(q, Xn) @ q['W'][2].T + q['B'][2], axis=1)


def predict_int8(q, Xn):
    return np.argmax(hidden_int8(q, Xn) @ q['QW'][2].astype(np.int32).T + q['QB'][2], axis=1)


def model_sizes(q):
    n_w = sum(w.size for w in q['W'])
    n_b = sum(b.size for b in q['B'])
    return (n_w + n_b) * 4, n_w + 4 * n_b