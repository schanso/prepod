import numpy as np

from prepod.lib.io import return_fnames, parse_raw, import_targets
from prepod.lib.constants import (COLNAME_SUBJID_SUDOCU, COLNAME_TARGET_SUDOCU,
                                  EXCLUDE_SUBJ)
from prepod.lib.models import train_test_wyrm, lda_vyrm, svm
from prepod.lib.prep import (align_bis, to_feature_vector, append_label,
                             merge_subjects, split_into_wins, filter_raw)


# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_log = path_data + 'info/log.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw/'
dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
dir_bis = path_data + 'rec/sudocu/bis/'
dir_out_raw = dir_raw + 'npy/frontal/'
dir_out_filtered = dir_filtered + 'frontal/'
dir_signal = dir_out_filtered


# INFO

fnames_raw = return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in EXCLUDE_SUBJ]


# PARAMS

test_size = .5
win_length = 5
bis_crit = 50
keep_proportion = 0.5
shrink = False
l_cutoff, h_cutoff = (8, 13)


# PARSE RAW FILES, FILTER, STORE AS NPY

subj_ids_subset = ['2153', '2170', '2291', '2324']
for subj_id in subj_ids:
    path_in = [dir_raw + el for el in fnames_raw if subj_id in el]
    path_out = '{}{}_L{}H{}K{}.npy'.format(
        *[dir_out_filtered, subj_id, l_cutoff, h_cutoff, keep_proportion]
    )
    data = parse_raw(path_in=path_in, ftype='edf', region='frontal')
    filtered = filter_raw(data,
                          srate=data.fs,
                          l_cutoff=l_cutoff,
                          h_cutoff=h_cutoff)
    np.save(path_out, arr=filtered)
    print('Successfully wrote data to ' + path_out)


# LOAD SUBJ DATA, APPEND LABELS, MERGE

path_out_merged = dir_out_filtered + 'merged.npy'
datasets = []
for subj_id in subj_ids_subset:
    path_signal = dir_signal + return_fnames(
        dir_in=dir_signal, substr=subj_id
    )[0]
    path_bis = dir_bis + subj_id + '/'
    data, bis = align_bis(path_signal=path_signal, path_bis=path_bis)
    data, bis = split_into_wins(
        data=data,
        bis_values=bis,
        win_length=win_length,
        bis_crit=bis_crit,
        keep_proportion=keep_proportion
    )
    label = import_targets(
        fpath=path_labels,
        colname_subjid=COLNAME_SUBJID_SUDOCU,
        colname_target=COLNAME_TARGET_SUDOCU,
        subj_ids=subj_id
    )
    data = to_feature_vector(data)
    data = append_label(data, label)
    datasets.append(data)
merge_subjects(datasets, path_out=path_out_merged)


# CLASSIFICATION (VANILLA LDA + SVM)

data = np.load(file=path_out_merged).flatten()[0]
tot = {'lda': [], 'svm': []}
for i in range(10):
    data_train, data_test = train_test_wyrm(data, test_size=test_size)
    acc_lda = lda_vyrm(data_train=data_train, data_test=data_test, shrink=shrink)
    acc_svm = svm(data_train, data_test, kernel='linear')

    tot['lda'].append(acc_lda)
    tot['svm'].append(acc_svm)

    print('Run {}\nLDA: {}, SVM: {} (train: {}, test: {})'.format(
            str(i+1), str(acc_lda), str(acc_svm), str(data_train.data.shape),
            str(data_test.data.shape)))

print('Mean/STD\nLDA: {}/{}, SVM: {}/{}'.format(
    str(np.mean(tot['lda'])), str(np.std(tot['lda'])),
    str(np.mean(tot['svm'])), str(np.std(tot['svm']))))

