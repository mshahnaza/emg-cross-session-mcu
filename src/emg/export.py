import os
import numpy as np
from .config import SEED, MAX_TEST, TEST_FROM_REP, WIN, FT_LR, FT_EPOCHS
from .models import zfit, zapply
from .calibration import run_methods, cal_order


def c_array(ctype, name, a):
    flat = np.asarray(a).ravel()
    if ctype == 'float':
        body = ', '.join(f'{v:.8e}f' for v in flat)
    else:
        body = ', '.join(str(int(v)) for v in flat)
    return f'const {ctype} {name}[{flat.size}] = {{{body}}};\n'


def write_header(path, q, pool, session_data, extra):
    X, y, r = session_data['X'], session_data['y'], session_data['rep']
    cal_X, cal_y = X[r <= 1], y[r <= 1]
    test_idx = np.where(r >= TEST_FROM_REP)[0]
    if len(test_idx) > MAX_TEST:
        test_idx = np.sort(np.random.default_rng(SEED).choice(test_idx, MAX_TEST, replace=False))
    test_X, test_y = X[test_idx], y[test_idx]
    mu_c, sd_c = zfit(cal_X)
    exp = run_methods(q, zapply(test_X, *pool), zapply(cal_X, mu_c, sd_c), cal_y, zapply(test_X, mu_c, sd_c), test_y)

    W, B = q['W'], q['B']
    h = '#pragma once\n'
    h += f'#define N_FEAT {W[0].shape[1]}\n#define N_CH {len(extra["channels"])}\n#define WIN {WIN}\n'
    h += f'#define H1 {W[0].shape[0]}\n#define H2 {W[1].shape[0]}\n#define N_CLASS {W[2].shape[0]}\n'
    h += f'#define N_CAL {len(cal_X)}\n#define N_TEST {len(test_X)}\n'
    h += f'#define EPOCHS {FT_EPOCHS}\n#define LR {float(np.float32(FT_LR))}f\n'
    for k, (af, ai) in enumerate(exp.values()):
        h += f'#define EXP_F{k} {af:.2f}f\n#define EXP_I{k} {ai:.2f}f\n'
    h += c_array('float', 'TRAIN_MU', pool[0]) + c_array('float', 'TRAIN_SD', pool[1])
    for i in range(3):
        h += c_array('float', f'W{i}', W[i]) + c_array('float', f'B{i}', B[i])
        h += c_array('int8_t', f'QW{i}', q['QW'][i]) + c_array('int32_t', f'QB{i}', q['QB'][i])
    h += c_array('float', 'S_ACT', q['S_ACT']) + c_array('float', 'S_MUL', q['S_MUL'])
    h += c_array('float', 'CAL_X', cal_X) + c_array('uint8_t', 'CAL_Y', cal_y) + c_array('uint16_t', 'CAL_ORDER', cal_order(len(cal_X)))
    h += c_array('float', 'TEST_X', test_X) + c_array('uint8_t', 'TEST_Y', test_y)
    h += c_array('float', 'DEMO_WINDOW', extra['demo_window']) + c_array('float', 'DEMO_FEATURES', extra['demo_features'])
    with open(path, 'w') as f:
        f.write(h)
    print('Header written:', path, round(os.path.getsize(path) / 1024), 'KB')
    return exp
