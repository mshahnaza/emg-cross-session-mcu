import pandas as pd
from scipy.stats import wilcoxon
from .experiments import rq1, train_calibration_models, renorm_table, board_methods_table
from .quantization import quantize_mlp

RANDOM = 'Random split (one session)'
WITHIN = 'Within session (reps 1-8 / 9-12)'
CROSS = 'Cross session (sessions 1-5 / 6-10)'
RENORM = 'Re-normalization'
PROTO = 'Re-norm + prototypes'
FT = 'Re-norm + last-layer fine-tuning'
ACC = ['No calibration', RENORM, 'Retrained']


def run_subject(data, subject):
    rq1_table, standard, pool = rq1(data, subject)
    models, X_sess = train_calibration_models(data, subject)
    renorm_df, _ = renorm_table(data, subject, standard, models, pool)
    q = quantize_mlp(models['MLP'], X_sess)
    board_df, _ = board_methods_table(data, subject, q, pool)
    for df in (rq1_table, renorm_df, board_df):
        df.insert(0, 'Subject', subject)
    return rq1_table, renorm_df, board_df


def mean_std(df, keys, values):
    g = df.groupby(keys)[values]
    out = g.mean().round(2)
    std = g.std().round(2)
    for v in values:
        out[v + ' std'] = std[v]
    return out


def wilcoxon_pairs(wide, pairs):
    rows = []
    for a, b in pairs:
        _, p = wilcoxon(wide[a], wide[b], zero_method='zsplit')
        rows.append([a, b, round(wide[a].mean(), 2), round(wide[b].mean(), 2), round((wide[b] - wide[a]).mean(), 2), round(p, 4)])
    return pd.DataFrame(rows, columns=['A', 'B', 'Mean A', 'Mean B', 'Mean B - A', 'p-value'])


def analyse(rq1_all, renorm_all, board_all):
    out = {}
    out['rq1_summary'] = mean_std(rq1_all, ['Protocol', 'Model'], ['Accuracy (%)'])
    rq1_wide = rq1_all.pivot_table(index=['Subject', 'Model'], columns='Protocol', values='Accuracy (%)').reset_index()
    out['rq1_tests'] = pd.concat([wilcoxon_pairs(rq1_wide[rq1_wide['Model'] == m], [(RANDOM, CROSS), (WITHIN, CROSS)]).assign(Model=m)
                                  for m in ['LDA', 'MLP']], ignore_index=True)

    renorm_subj = renorm_all.groupby(['Subject', 'Model', 'Budget (reps)'])[ACC].mean().reset_index()
    summary = mean_std(renorm_subj, ['Model', 'Budget (reps)'], ACC)
    summary['R'] = ((summary[RENORM] - summary['No calibration']) / (summary['Retrained'] - summary['No calibration'])).round(2)
    out['renorm_summary'] = summary
    out['renorm_tests'] = pd.concat([wilcoxon_pairs(part, [('No calibration', RENORM)]).assign(Model=m, Budget=b)
                                     for (m, b), part in renorm_subj.groupby(['Model', 'Budget (reps)'])], ignore_index=True)

    board_subj = board_all.groupby(['Subject', 'Budget (reps)', 'Method'])[['Float32', 'INT8']].mean().reset_index()
    out['board_summary'] = mean_std(board_subj, ['Method', 'Budget (reps)'], ['Float32', 'INT8'])
    base = renorm_subj[renorm_subj['Model'] == 'MLP'][['Subject', 'Budget (reps)', 'No calibration', 'Retrained']]
    base = base.rename(columns={'No calibration': 'Standard model'})
    wide = board_subj.pivot_table(index=['Subject', 'Budget (reps)'], columns='Method', values='Float32').reset_index()
    wide = wide.merge(base, on=['Subject', 'Budget (reps)'])
    pairs = [('Standard model', RENORM), ('Standard model', FT), (RENORM, FT), (RENORM, PROTO), ('Retrained', FT)]
    out['method_tests'] = pd.concat([wilcoxon_pairs(part, pairs).assign(Budget=b) for b, part in wide.groupby('Budget (reps)')],
                                    ignore_index=True)
    means = wide.groupby('Budget (reps)').mean(numeric_only=True)
    out['recovered_fraction'] = pd.DataFrame({m: (means[m] - means['Standard model']) / (means['Retrained'] - means['Standard model'])
                                              for m in [RENORM, PROTO, FT]}).round(2)
    out['int8_tests'] = pd.concat([wilcoxon_pairs(part, [('Float32', 'INT8')]).assign(Method=m, Budget=b)
                                   for (m, b), part in board_subj.groupby(['Method', 'Budget (reps)'])], ignore_index=True)
    return out
