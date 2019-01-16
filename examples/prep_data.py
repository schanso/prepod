import os

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.prep as prep


# PARAMS

study = 'Sudocu'
regions = ['fronto-parietal', 'frontal', 'full']
freq_bands = ['alpha', 'beta', 'below15', 'delta', 'theta']
win_lengths = [60, 30, 10, 5]

for region in regions:
    for freq_band in freq_bands:
        lcut, hcut = const.FREQ_BANDS[freq_band]


        # PATHS

        path_data = '/Users/jannes/Projects/delir/data/'
        path_labels = path_data + 'info/sudocu_info/subject_data.csv'
        dir_raw = path_data + 'rec/sudocu/brainvision/raw'
        dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
        if not os.path.exists(dir_filtered):
            os.makedirs(dir_filtered)
        dir_out_filtered = dir_filtered + region
        if not os.path.exists(dir_out_filtered):
            os.makedirs(dir_out_filtered)


        # INFO

        fnames_raw = hlp.return_fnames(dir_in=dir_raw)
        subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
        subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]


        # PARSE RAW FILES, FILTER, STORE AS NPY

        for subj_id in subj_ids:
            path_in = [dir_raw + '/' + el for el in fnames_raw if subj_id in el]
            path_out = '{}/{}/{}.npy'.format(dir_out_filtered, freq_band, subj_id)
            data = io.parse_raw(path_in=path_in, ftype='edf', region=region)
            filtered = prep.filter_raw(data, srate=data.fs, l_cutoff=lcut, h_cutoff=hcut, b_pass=True)
            io.save_as_npy(data=filtered, path_out=path_out)


# LOAD SUBJ DATA, APPEND LABELS, MERGE

for win_length in win_lengths:
    for freq_band in freq_bands:
        lcut, hcut = const.FREQ_BANDS[freq_band]
        for region in regions:

            # PATHS

            path_data = '/Users/jannes/Projects/delir/data/'
            path_labels = path_data + 'info/sudocu_info/subject_data.csv'
            dir_raw = path_data + 'rec/sudocu/brainvision/raw'
            dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
            if not os.path.exists(dir_filtered):
                os.makedirs(dir_filtered)
            dir_bis = path_data + 'rec/sudocu/bis/'
            dir_out_filtered = dir_filtered + region
            if not os.path.exists(dir_out_filtered):
                os.makedirs(dir_out_filtered)
            dir_signal = '{}/{}'.format(dir_out_filtered, freq_band)
            if not os.path.exists(dir_signal):
                os.makedirs(dir_signal)
            fname_merged = 'complete_{}s.npy'.format(win_length)
            path_out_merged = '{}/{}'.format(dir_signal, fname_merged)

            # INFO

            fnames_raw = hlp.return_fnames(dir_in=dir_raw)
            subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
            subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]

            datasets = []
            for subj_id in subj_ids:
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
            prep.merge_subjects(datasets, path_out=path_out_merged)

