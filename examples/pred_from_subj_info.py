import pandas as pd
from sklearn.linear_model import LogisticRegression as logreg
from sklearn import svm

import prepod.lib.io as io
import prepod.lib.prep as prep
import prepod.lib.models as mdl

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_out = '/Users/jannes/Projects/delir/results/test_{}.csv'
target = 'delir_60min'
data = io.parse_subj_info(path_labels, 'Sudocu')
data = prep.drop_non_feature_cols(data, target)
data = prep.drop_if_too_many_nans(data, .25)
features = list(data.drop(['subj_id', 'delir_60min', 'age'], axis=1))
data = prep.drop_na(data, features)
data = prep.to_fv(data, features, target)

clfs = [svm.SVC(gamma='scale'), logreg(solver='liblinear')]
for clf in clfs:
    res = mdl.backward_subset_selection(data['X'], data['y'], data['X_labels'], data['y_labels'], K=1, clf=clf)
    res = pd.DataFrame(res).sort_values(by='mean_acc', ascending=False)
    res = res[res['n_features'] < 10]
    res['target'] = target
    with open(path_out.format('bss'), 'a') as f:
        res.to_csv(f, index=False)
    res = mdl.forward_subset_selection(data, K=len(features), init_combos=2, clf=clf)
    res = pd.DataFrame(res).sort_values(by='mean_acc', ascending=False)
    res = res[res['n_features'] < 10]
    res['target'] = target
    with open(path_out.format('fss'), 'a') as f:
        res.to_csv(f, header=False, index=False)

