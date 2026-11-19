FS = 2000
WIN = int(0.200 * FS)
STEP = int(0.050 * FS)
GESTURES = [1, 3, 4, 6, 9, 10, 11]
SEED = 0
BUDGETS = [1, 2, 4]
TEST_FROM_REP = 5
FT_LR = 0.005
FT_EPOCHS = 20
MAX_TEST = 700


def sessions(subject):
    return [f'S{subject}_D{d}_T{t}.mat' for d in range(1, 6) for t in (1, 2)]


def train_sessions(subject):
    return sessions(subject)[:5]


def test_sessions(subject):
    return sessions(subject)[5:]