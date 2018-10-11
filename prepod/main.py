import numpy as np

from prepod.lib.io import return_fnames, parse_raw, import_targets
from prepod.lib.constants import (COLNAME_SUBJID_SUDOCU, COLNAME_TARGET_SUDOCU,
                                  EXCLUDE_SUBJ)
from prepod.lib.models import train_test_wyrm, lda_vyrm
from prepod.lib.prep import (align_bis, to_feature_vector, append_label,
                             merge_subjects)


# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw/'
dir_bis = path_data + 'rec/sudocu/bis/'
dir_out = dir_raw + 'npy/frontal/'
dir_signal = dir_out
path_out_merged = dir_out + 'merged.npy'


# INFO

fnames_raw = return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in EXCLUDE_SUBJ]

# PARSE RAW FILES AND STORE AS NPY

for subj_id in subj_ids:
    path = [dir_raw + el for el in fnames_raw if subj_id in el]
    parse_raw(path_in=path, dir_out=dir_out, ftype='edf', region='frontal')


# LOAD SUBJ DATA, APPEND LABELS, MERGE

subj_ids_subset = ['2100', '2129', '2137', '2153', '2156', '2199']
datasets = []
for subj_id in subj_ids_subset:
    path_signal = dir_signal + return_fnames(dir_in=dir_signal, substr=subj_id)
    path_bis = dir_bis + subj_id + '/'
    data, bis = align_bis(path_signal=path_signal, path_bis=path_bis)
    label = import_targets(fpath=path_labels,
                           colname_subjid=COLNAME_SUBJID_SUDOCU,
                           colname_target=COLNAME_TARGET_SUDOCU,
                           subj_ids=subj_id)
    data = to_feature_vector(data)
    data = append_label(data, label)
    datasets.append(data)
merge_subjects(datasets, path_out=path_out_merged)


# CLASSIFICATION (VANILLA LDA)

data = np.load(file=path_out_merged).flatten()[0]
data_train, data_test = train_test_wyrm(data, test_size=.5)
acc = lda_vyrm(data_train=data_train, data_test=data_test)
print(acc)

