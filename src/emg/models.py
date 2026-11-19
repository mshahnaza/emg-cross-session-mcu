import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from .config import SEED


def zfit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd


def zapply(X, mu, sd):
    return (X - mu) / sd


def new_mlp():
    return MLPClassifier(hidden_layer_sizes=(16, 32), activation='relu', solver='adam',
                         learning_rate_init=1e-3, batch_size=32, max_iter=150, random_state=SEED)


def new_lda():
    return LinearDiscriminantAnalysis()


def new_model(name):
    return new_lda() if name == 'LDA' else new_mlp()


def acc(model, X, y):
    return float((model.predict(X) == y).mean() * 100)


def stack(data, names, key):
    return np.concatenate([data[s][key] for s in names])