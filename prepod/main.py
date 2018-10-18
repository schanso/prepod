import os

import numpy as np

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.models as models
import prepod.lib.prep as prep


# PARAMS

study = 'Sudocu'
region = 'frontal'
freq_band = 'alpha'
lcut, hcut = const.FREQ_BANDS[freq_band]

test_size = .5
win_length = 5
bis_crit = 60
drop_perc = .66
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

fnames_raw = hlp.return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]
subj_ids = [el for el in subj_ids if int(el) <= 2262]


# PARSE RAW FILES, FILTER, STORE AS NPY

for subj_id in subj_ids:
    path_in = [dir_raw + '/' + el for el in fnames_raw if subj_id in el]
    path_out = '{}/{}/{}.npy'.format(dir_out_filtered, freq_band, subj_id)
    data = io.parse_raw(path_in=path_in, ftype='edf', region=region)
    filtered = prep.filter_raw(data, srate=data.fs, l_cutoff=lcut, h_cutoff=hcut)
    io.save_as_npy(data=filtered, path=path_out)


# LOAD SUBJ DATA, APPEND LABELS, MERGE

datasets = []
for subj_id in subj_ids:
    curr_fname = hlp.return_fnames(dir_in=dir_signal, substr=subj_id)
    path_signal = '{}/{}'.format(dir_signal, hlp.return_fnames(dir_in=dir_signal, substr=subj_id))
    path_bis = dir_bis + subj_id + '/'
    data = io.load_wyrm(path=path_signal)
    data.markers = prep.create_markers(data, win_length)
    data.label = prep.fetch_labels(path_labels, study, subj_id)
    data = prep.match_bis(data, path_bis)
    data = prep.segment_data(data, win_length)
    data.subj_id = prep.append_subj_id(data, subj_id)
    datasets.append(data)
prep.merge_subjects(datasets, path_out=path_out_merged)


# CLASSIFICATION (VANILLA LDA + SVM)

data = io.load_wyrm(path=path_out_merged)
data = prep.subset_data(data, bis_crit=bis_crit, drop_perc=drop_perc, drop_from='end')
data = prep.apply_csp(data)
data = prep.create_fvs(data)
tot = {'lda': [], 'svm': []}
for i in range(len(subj_ids)):
    data_train, data_test = models.train_test_cv(data, leave_out=i)
    acc_lda = models.lda(data_train=data_train, data_test=data_test, solver=solver, shrinkage=shrink)
    acc_svm = models.svm(data_train, data_test, kernel='linear')

    tot['lda'].append(acc_lda)
    tot['svm'].append(acc_svm)

    print('Run {}\nLDA: {}, SVM: {} (train: {}, test: {}\nLeft out: {})'.format(
            str(i+1), str(acc_lda), str(acc_svm), str(data_train.data.shape),
            str(data_test.data.shape), str(np.unique(data_test.subj_id))))

print('Mean/STD\nLDA: {}/{}, SVM: {}/{}'.format(
    str(np.mean(tot['lda'])), str(np.std(tot['lda'])),
    str(np.mean(tot['svm'])), str(np.std(tot['svm']))))

