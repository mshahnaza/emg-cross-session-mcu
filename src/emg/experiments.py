import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import SEED, BUDGETS, TEST_FROM_REP, train_sessions, test_sessions
from .models import zfit, zapply, new_model, acc, stack
from .calibration import run_methods


def rq1(data, subject):
    rows = []
    s0 = train_sessions(subject)[0]
    X, y = data[s0]['X'], data[s0]['y']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    mu, sd = zfit(Xtr)
    for name in ['LDA', 'MLP']:
        m = new_model(name).fit(zapply(Xtr, mu, sd), ytr)
        rows.append(['Random split (one session)', name, acc(m, zapply(Xte, mu, sd), yte)])
    tr = data[s0]['rep'] <= 8
    mu, sd = zfit(X[tr])
    for name in ['LDA', 'MLP']:
        m = new_model(name).fit(zapply(X[tr], mu, sd), y[tr])
        rows.append(['Within session (reps 1-8 / 9-12)', name, acc(m, zapply(X[~tr], mu, sd), y[~tr])])
    Xtr, ytr = stack(data, train_sessions(subject), 'X'), stack(data, train_sessions(subject), 'y')
    mu, sd = zfit(Xtr)
    standard = {}
    for name in ['LDA', 'MLP']:
        m = new_model(name).fit(zapply(Xtr, mu, sd), ytr)
        standard[name] = m
        a = [acc(m, zapply(data[s]['X'], mu, sd), data[s]['y']) for s in test_sessions(subject)]
        rows.append(['Cross session (sessions 1-5 / 6-10)', name, float(np.mean(a))])
    table = pd.DataFrame(rows, columns=['Protocol', 'Model', 'Accuracy (%)']).round(1)
    return table, standard, (mu, sd)


def train_calibration_models(data, subject):
    names = train_sessions(subject)
    X_sess = np.concatenate([zapply(data[s]['X'], *zfit(data[s]['X'])) for s in names])
    y = stack(data, names, 'y')
    models = {name: new_model(name).fit(X_sess, y) for name in ['LDA', 'MLP']}
    return models, X_sess


def renorm_table(data, subject, standard, models, pool):
    rows = []
    for s in test_sessions(subject):
        X, y, r = data[s]['X'], data[s]['y'], data[s]['rep']
        test = r >= TEST_FROM_REP
        for name, m in models.items():
            none = acc(standard[name], zapply(X[test], *pool), y[test])
            mu_r, sd_r = zfit(X[r <= 4])
            ret = new_model(name).fit(zapply(X[r <= 4], mu_r, sd_r), y[r <= 4])
            full = acc(ret, zapply(X[test], mu_r, sd_r), y[test])
            for b in BUDGETS:
                mu_c, sd_c = zfit(X[r <= b])
                cal = acc(m, zapply(X[test], mu_c, sd_c), y[test])
                R = (cal - none) / (full - none) if full != none else np.nan
                rows.append([s, name, b, none, cal, full, R])
    df = pd.DataFrame(rows, columns=['Session', 'Model', 'Budget (reps)', 'No calibration', 'Re-normalization', 'Retrained', 'R'])
    summary = df.groupby(['Model', 'Budget (reps)'])[['No calibration', 'Re-normalization', 'Retrained', 'R']].mean().round(2)
    return df, summary


def board_methods_table(data, subject, q, pool):
    rows = []
    for s in test_sessions(subject):
        X, y, r = data[s]['X'], data[s]['y'], data[s]['rep']
        test = r >= TEST_FROM_REP
        for b in BUDGETS:
            cal = r <= b
            mu_c, sd_c = zfit(X[cal])
            res = run_methods(q, zapply(X[test], *pool), zapply(X[cal], mu_c, sd_c), y[cal],
                              zapply(X[test], mu_c, sd_c), y[test])
            for method, (af, ai) in res.items():
                rows.append([s, b, method, af, ai])
    df = pd.DataFrame(rows, columns=['Session', 'Budget (reps)', 'Method', 'Float32', 'INT8'])
    summary = df.groupby(['Method', 'Budget (reps)'])[['Float32', 'INT8']].mean().round(2)
    return df, summary