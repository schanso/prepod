from prepod.lib.io import return_fnames, parse_raw, import_targets
from prepod.lib.constants import COLNAME_SUBJID_SUDOCU, COLNAME_TARGET_SUDOCU
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


# PARSE RAW FILES AND STORE AS NPY

for subj_id in subj_ids:
    path = [dir_raw + el for el in fnames_raw if subj_id in el]
    parse_raw(path_in=path, dir_out=dir_out, ftype='edf', region='frontal')


# LOAD SUBJ DATA, APPEND LABELS, MERGE

datasets = []
for subj_id in subj_ids:
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
data = merge_subjects(datasets, path_out=path_out_merged)

