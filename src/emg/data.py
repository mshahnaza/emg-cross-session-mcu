import os
import time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from .config import FS, WIN, STEP, GESTURES, sessions


def load_mat(path):
    try:
        d = sio.loadmat(path)
    except NotImplementedError:
        import mat73
        d = mat73.loadmat(path)
    emg = np.asarray(d['emg'], dtype=np.float64)
    lab = np.asarray(d['restimulus']).ravel().astype(int)
    rep = np.asarray(d['rerepetition']).ravel().astype(int)
    return emg, lab, rep


def make_filter():
    sos_bp = butter(4, [20, 450], btype='bandpass', fs=FS, output='sos')
    b, a = iirnotch(50, Q=30, fs=FS)
    return np.vstack([sos_bp, tf2sos(b, a)])


SOS = make_filter()


def window_sums(x, starts, length):
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    return c[starts + length] - c[starts]


def extract(emg, lab, rep):
    x = sosfilt(SOS, emg, axis=0)
    starts = np.arange(0, len(x) - WIN + 1, STEP)
    mav = window_sums(np.abs(x), starts, WIN) / WIN
    rms = np.sqrt(window_sums(x ** 2, starts, WIN) / WIN)
    d = np.vstack([np.zeros((1, x.shape[1])), np.abs(np.diff(x, axis=0))])
    wl = window_sums(d, starts + 1, WIN - 1)
    z = np.vstack([np.zeros((1, x.shape[1])), (x[1:] * x[:-1] < 0).astype(float)])
    zc = window_sums(z, starts + 1, WIN - 1)
    lab_change = np.concatenate([[0], (np.diff(lab) != 0).astype(float)])[:, None]
    rep_change = np.concatenate([[0], (np.diff(rep) != 0).astype(float)])[:, None]
    stable = (window_sums(lab_change, starts + 1, WIN - 1)[:, 0] == 0) & (window_sums(rep_change, starts + 1, WIN - 1)[:, 0] == 0)
    y = lab[starts]
    r = rep[starts]
    keep = stable & np.isin(y, GESTURES)
    X = np.hstack([mav, rms, wl, zc])[keep]
    y = np.array([GESTURES.index(g) for g in y[keep]])
    return X.astype(np.float32), y, r[keep], x, starts[keep]


def extract_subject(folder, subject):
    names = sessions(subject)
    first, _, _ = load_mat(os.path.join(folder, names[0]))
    channels = np.where(np.abs(first).sum(axis=0) > 0)[0]
    del first
    out = {'channels': channels}
    for i, name in enumerate(names):
        t0 = time.time()
        emg, lab, rep = load_mat(os.path.join(folder, name))
        X, y, r, xf, st = extract(emg[:, channels], lab, rep)
        out[f'X_{i}'], out[f'y_{i}'], out[f'rep_{i}'] = X, y, r
        if i == len(names) - 1:
            k = np.where(r >= 5)[0][0]
            out['demo_window'] = xf[st[k]: st[k] + WIN].astype(np.float32)
            out['demo_features'] = X[k]
        del emg, xf
        print(f'{name}: {X.shape[0]} windows, {time.time() - t0:.1f} s')
    return out


def save_features(features, path):
    np.savez_compressed(path, **features)


def load_features(path, subject):
    f = np.load(path)
    data = {}
    for i, name in enumerate(sessions(subject)):
        data[name] = {'X': f[f'X_{i}'], 'y': f[f'y_{i}'], 'rep': f[f'rep_{i}']}
    extra = {'channels': f['channels'], 'demo_window': f['demo_window'], 'demo_features': f['demo_features']}
    return data, extra