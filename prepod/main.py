import numpy as np
import os

import prepod.lib.prep as prep
from prepod.lib.io import return_fnames, parse_raw, import_targets
from prepod.lib.constants import (COLNAME_SUBJID_SUDOCU, COLNAME_TARGET_SUDOCU,
                                  EXCLUDE_SUBJ)
import prepod.lib.constants as const
from prepod.lib.models import train_test_wyrm, lda_vyrm, svm, train_test_cv, lda
from prepod.lib.prep import (align_bis,
                             merge_subjects, split_into_wins, filter_raw, subset_data,
                             calc_csp)


# PARAMS

study = 'Sudocu'
region = 'frontal'
freq_band = 'alpha'
l_cutoff, h_cutoff = const.FREQ_BANDS[freq_band]

test_size = .5
win_length = 5
bis_crit = 50
drop_perc = .5
drop_from = 'beginning'
solver = 'svd'
shrink = False


# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_log = path_data + 'info/log.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw'
dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
if not os.path.exists(dir_filtered):
    os.makedirs(dir_filtered)
dir_bis = path_data + 'rec/sudocu/bis/'
dir_out_raw = '{}/{}/{}'.format(dir_raw, 'npy', region)
dir_out_filtered = dir_filtered + region
if not os.path.exists(dir_out_filtered):
    os.makedirs(dir_out_filtered)
dir_signal = '{}/{}'.format(dir_out_filtered, freq_band)
if not os.path.exists(dir_signal):
    os.makedirs(dir_signal)
fname_merged = 'complete.npy'
path_out_merged = '{}/{}'.format(dir_signal, fname_merged)


# INFO

fnames_raw = return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in EXCLUDE_SUBJ]
subj_ids = [el for el in subj_ids if int(el) <= 2262]


# PARSE RAW FILES, FILTER, STORE AS NPY

for subj_id in subj_ids:
    path_in = [dir_raw + '/' + el for el in fnames_raw if subj_id in el]
    path_out = '{}/{}/{}.npy'.format(dir_out_filtered, freq_band, subj_id)
    data = parse_raw(path_in=path_in, ftype='edf', region=region)
    filtered = filter_raw(data,
                          srate=data.fs,
                          l_cutoff=l_cutoff,
                          h_cutoff=h_cutoff)
    np.save(path_out, arr=filtered)
    print('Successfully wrote data to ' + path_out)


# # LOAD SUBJ DATA, APPEND LABELS, MERGE

datasets = []
for subj_id in subj_ids:
    curr_fname = return_fnames(dir_in=dir_signal, substr=subj_id)
    path_signal = '{}/{}'.format(dir_signal, return_fnames(dir_in=dir_signal, substr=subj_id))
    path_bis = dir_bis + subj_id + '/'
    data = np.load(file=path_signal).flatten()[0]
    data.markers = prep.create_markers(data, win_length)
    data.label = prep.fetch_labels(path_labels, study, subj_id)
    data = prep.match_bis(data, path_bis)
    data = prep.segment_data(data, win_length)
    data.subj_id = prep.append_subj_id(data, subj_id)
    datasets.append(data)
merge_subjects(datasets, path_out=path_out_merged)


# CLASSIFICATION (VANILLA LDA + SVM)

data = np.load(file=path_out_merged).flatten()[0]
print('Done loading')
data = prep.subset_data(data, bis_crit=bis_crit, drop_perc=drop_perc, drop_from='end')
data = prep.create_fvs(data)
tot = {'lda': [], 'svm': []}
for i in range(len(subj_ids)):
    data_train, data_test = train_test_cv(data, counter=i)
    acc_lda = lda(data_train=data_train, data_test=data_test, solver=solver, shrinkage=shrink)
    acc_svm = svm(data_train, data_test, kernel='linear')

    tot['lda'].append(acc_lda)
    tot['svm'].append(acc_svm)

    print('Run {}\nLDA: {}, SVM: {} (train: {}, test: {})'.format(
            str(i+1), str(acc_lda), str(acc_svm), str(data_train.data.shape),
            str(data_test.data.shape)))

print('Mean/STD\nLDA: {}/{}, SVM: {}/{}'.format(
    str(np.mean(tot['lda'])), str(np.std(tot['lda'])),
    str(np.mean(tot['svm'])), str(np.std(tot['svm']))))

