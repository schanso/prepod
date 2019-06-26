import os
import datetime

import numpy as np
import pandas as pd

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.models as models
import prepod.lib.prep as prep

# PARAMS

regions = ['fronto-parietal', 'frontal']
freq_bands = ['slow', 'delta', 'alpha', 'beta', 'below15']
bis_crits = [100, 90, 80, 70, 60, 50, 40, 30]
study = 'Sudocu'
test_size = .5
n_leave_out = 2
win_lengths = [60, 30]
drop_perc = None  # [.95, .85, .75, .65, .50, .33, .25, .10]
drop_from = None  # 'beginning'
solver = 'svd'
shrink = False
kernel = 'rbf'
include_external = False
use_mins = [(60, 45), (45, 30), (30, 15), (15, 0)]
use_froms = ['beginning', 'end']

# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_log = path_data + 'info/log.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw'
dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
if not os.path.exists(dir_filtered):
    os.makedirs(dir_filtered)
dir_bis = path_data + 'rec/sudocu/bis/'

# INFO

fnames_raw = hlp.return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]


for region in regions:
    dir_out_raw = '{}/{}/{}'.format(dir_raw, 'npy', region)
    dir_out_filtered = dir_filtered + region
    if not os.path.exists(dir_out_filtered):
        os.makedirs(dir_out_filtered)

    for freq_band in freq_bands:
        lcut, hcut = const.FREQ_BANDS[freq_band]
        dir_signal = '{}/{}'.format(dir_out_filtered, freq_band)
        if not os.path.exists(dir_signal):
            os.makedirs(dir_signal)
        for win_length in win_lengths:
            fname_merged = 'complete_{}s.npy'.format(win_length)
            path_out_merged = '{}/{}'.format(dir_signal, fname_merged)

            # PARSE RAW FILES, FILTER, STORE AS NPY

            for subj_id in subj_ids:
                path_in = [dir_raw + '/' + el for el in fnames_raw if subj_id in el]
                path_out = '{}/{}/{}.npy'.format(dir_out_filtered, freq_band, subj_id)
                if not os.path.exists(path_out):
                    data = io.parse_raw(path_in=path_in, ftype='edf', region=region)
                    filtered = prep.filter_raw(data, srate=data.fs, l_cutoff=lcut, h_cutoff=hcut, b_pass=True)
                    io.save_as_npy(data=filtered, path_out=path_out)
                # plot.plot_raw_vs_filt(data, filtered, n_sec=30, t0=600)


            # LOAD SUBJ DATA, APPEND LABELS, MERGE
            _data = None
            try:
                _data = io.load_pickled(path_in=path_out_merged)
            except Exception:
                datasets = []
                for subj_id in subj_ids[:10]:
                    curr_fname = hlp.return_fnames(dir_in=dir_signal, substr=subj_id)
                    path_signal = '{}/{}'.format(dir_signal, hlp.return_fnames(dir_in=dir_signal, substr=subj_id))
                    path_bis = dir_bis + subj_id + '/'
                    data = io.load_wyrm(path_in=path_signal)
                    data.markers = prep.create_markers(data, win_length)
                    data.label = prep.fetch_labels(path_labels, study, subj_id)
                    data = prep.match_bis(data, path_bis)
                    data = prep.segment_data(data, win_length)
                    data.subj_id = prep.append_subj_id(data, subj_id)
                    datasets.append(data)
                _data = prep.merge_subjects(datasets)

            for bis_crit in bis_crits:
                for use_from in use_froms:
                    for use_min in use_mins:
                        data = prep.subset_data(_data, bis_crit=bis_crit, drop_perc=drop_perc, drop_from=drop_from, use_min=use_min, use_from=use_from)
                        x, acc_lda_all_runs, acc_svm_all_runs, acc_svm_lin_all_runs, acc_lda_eigen_all_runs = [], [], [], [], []
                        acc_lda_class0_all_runs, acc_lda_class1_all_runs = [], []
                        acc_svm_class0_all_runs, acc_svm_class1_all_runs = [], []
                        acc_svm_lin_class0_all_runs, acc_svm_lin_class1_all_runs = [], []
                        acc_lda_eigen_class0_all_runs, acc_lda_eigen_class1_all_runs = [], []
                        print('Start training on subjects: {}'.format(', '.join(subj_ids)))
                        for i in range(len(subj_ids)):
                            try:
                                data_train, data_test, left_out = models.train_test_cv(data, n_leave_out=n_leave_out, idx=i)
                                data_train = prep.apply_csp(data_train, return_as='logvar')
                                data_test = prep.apply_csp(data_test, return_as='logvar')
                                if include_external:
                                    data_train = prep.append_external_features(data_train, fpath=path_labels)
                                    data_test = prep.append_external_features(data_test, fpath=path_labels)
                                acc_lda, acc_lda_class0, acc_lda_class1 = models.lda(data_train=data_train, data_test=data_test, solver=solver, shrinkage=shrink, return_per_patient=True)
                                acc_svm, acc_svm_class0, acc_svm_class1 = models.svm(data_train, data_test, kernel=kernel, return_per_patient=True)
                                acc_svm_lin, acc_svm_lin_class0, acc_svm_lin_class1 = models.svm(data_train, data_test, kernel='linear', return_per_patient=True)
                                acc_lda_eigen, acc_lda_eigen_class0, acc_lda_eigen_class1 = models.lda(data_train=data_train, data_test=data_test, solver='eigen', shrinkage='auto', return_per_patient=True)
                                acc_lda_all_runs.append(acc_lda)
                                acc_lda_class0_all_runs.append(acc_lda_class0)
                                acc_lda_class1_all_runs.append(acc_lda_class1)
                                acc_svm_all_runs.append(acc_svm)
                                acc_svm_class0_all_runs.append(acc_svm_class0)
                                acc_svm_class1_all_runs.append(acc_svm_class1)
                                acc_svm_lin_all_runs.append(acc_svm_lin)
                                acc_svm_lin_class0_all_runs.append(acc_svm_lin_class0)
                                acc_svm_lin_class1_all_runs.append(acc_svm_lin_class1)
                                acc_lda_eigen_all_runs.append(acc_lda_eigen)
                                acc_lda_eigen_class0_all_runs.append(acc_lda_eigen_class0)
                                acc_lda_eigen_class1_all_runs.append(acc_lda_eigen_class1)
                                x.append(', '.join(left_out))

                                print('Run {}/{}: {:.3f} (LDA), {:.3f} (SVM), {:.3f} (SVM_lin), {:.3f} (SVM_eigen) (left out: {})'.format(
                                    i+1, len(subj_ids), acc_lda, acc_svm, acc_svm_lin , acc_lda_eigen, left_out))
                            except IndexError:
                                print('Not enough data for all other subjects. Will break.')
                                break

                        print('Mean over all runs:\n LDA: {} (std: {})\n SVM: {} (std: {})\n SVM_lin: {} (std: {})\n LDA_eigen: {} (std: {})'.format(
                            np.mean(acc_lda_all_runs), np.std(acc_lda_all_runs),
                            np.mean(acc_svm_all_runs), np.std(acc_svm_all_runs),
                            np.mean(acc_svm_lin_all_runs), np.std(acc_svm_lin_all_runs),
                            np.mean(acc_lda_eigen_all_runs), np.std(acc_lda_eigen_all_runs)
                        ))

                        d = {
                            'date': datetime.datetime.now(),
                            'mean_lda': np.mean(acc_lda_all_runs),
                            'std_lda': np.std(acc_lda_all_runs),
                            'mean_svm': np.mean(acc_svm_all_runs),
                            'std_svm': np.std(acc_svm_all_runs),
                            'region': region,
                            'freq_band': freq_band,
                            'test_size': test_size,
                            'n_leave_out': n_leave_out,
                            'win_length': win_length,
                            'bis_crit': bis_crit,
                            'drop_perc': drop_perc,
                            'drop_from': drop_from,
                            'solver': solver,
                            'shrink': shrink,
                            'kernel': kernel,
                            'include_external': include_external,
                            'external_factors': 'None',
                            'n_runs': len(acc_lda_all_runs),
                            'use_min': str(use_min),
                            'use_from': use_from,
                            'below_500_lda': len(np.where(np.array(acc_lda_all_runs) < .501)[0]),
                            'below_500_svm': len(np.where(np.array(acc_svm_all_runs) < .501)[0]),
                            'acc_lda_class0_all_runs': str(acc_lda_class0_all_runs),
                            'acc_lda_class1_all_runs': str(acc_lda_class1_all_runs),
                            'acc_svm_class0_all_runs': str(acc_svm_class0_all_runs),
                            'acc_svm_class1_all_runs': str(acc_svm_class1_all_runs),
                            'mean_svm_lin': np.mean(acc_svm_lin_all_runs),
                            'std_svm_lin': np.std(acc_svm_lin_all_runs),
                            'acc_svm_lin_class0_all_runs': str(acc_svm_lin_class0_all_runs),
                            'acc_svm_lin_class1_all_runs': str(acc_svm_lin_class1_all_runs),
                            'mean_lda_eigen': np.mean(acc_lda_eigen_all_runs),
                            'std_lda_eigen': np.std(acc_lda_eigen_all_runs),
                            'acc_lda_eigen_class0_all_runs': str(acc_lda_eigen_class0_all_runs),
                            'acc_lda_eigen_class1_all_runs': str(acc_lda_eigen_class1_all_runs)
                        }
                        cols = ['date', 'mean_lda', 'std_lda', 'mean_svm', 'std_svm', 'region', 'freq_band',
                                'test_size', 'n_leave_out', 'win_length', 'bis_crit', 'drop_perc', 'drop_from',
                                'solver', 'shrink', 'kernel', 'include_external', 'external_factors', 'n_runs',
                                'use_min', 'use_from', 'below_500_lda', 'below_500_svm', 'acc_lda_class0_all_runs',
                                'acc_lda_class1_all_runs', 'acc_svm_class0_all_runs', 'acc_svm_class1_all_runs',
                                'mean_svm_lin', 'std_svm_lin', 'acc_svm_lin_class0_all_runs', 'acc_svm_lin_class1_all_runs',
                                'mean_lda_eigen', 'std_lda_eigen', 'acc_lda_eigen_class0_all_runs', 'acc_lda_eigen_class1_all_runs']

                        df = pd.DataFrame(d, index=[0], columns=cols)
                        path = '/Users/jannes/Projects/delir/results/acc/res.csv'
                        with open(path, 'a') as f:
                            df.to_csv(f, header=False)
