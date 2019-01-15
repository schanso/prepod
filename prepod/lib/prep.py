import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter, iirnotch
from wyrm.processing import append as wyrm_append, segment_dat, select_epochs, calculate_csp, apply_spatial_filter
from wyrm.types import Data

import prepod.lib.constants as const
import prepod.lib.io as io
import prepod.lib.models as mdl


def detect_bads(x, srate, corr_threshold=0.4, window_width=1,
                perc_of_windows=0.01, set_to_zero=True):
    """Detects bad channels.

    Uses the deviation and correlation criterion introduced by
    Bigdely-Shamlo et al. (2015). For the deviation criterion, channels
    are marked as bad if the robust z-score of their SD is > 5. For the
    correlation criterion, the corr. between a channel and all others is
    calculated in non-overlapping time windows (1 s by default). Within
    each, the maximum absolute correlation is compared to a threshold
    (0.4 by default). If, for a given channel, the maximum absolute
    correlation is below threshold for a certain percentage of windows
    (1 % by default), the channel is marked as bad.

    Params
    ------
        srate : float
            sampling rate of the signal
        corr_threshold : float
            threshold for the maximum absolute correlation
        window_width : int
            width of window within which to compare channels (seconds)
        perc_of_windows : float
            between 0 and 1, percentage of windows within which a channel
            must be uncorrelated with other channels to be marked as bad
        return_indices : boolean
            whether to return a list of indices of bad channels

    Returns
    -------
        cleaned_raw : ndArray
            cleaned raw signal (channels x samples), bads dropped
        bads_idx : list
            list of indices of dropped bad channels (optional)

    --- IN DEVELOPMENT ---
    """
    # TODO: start, stop always floored; implement handling of bads
    if not (0 <= perc_of_windows <= 1):
        raise TypeError('Parameter perc_of_windows has to be decimal between '
                        '0 and 1.')

    # Deviation criterion
    std = np.std(x, axis=1)
    median = np.median(std)
    mad = np.median(abs(std - median))
    robust_z = abs(std - median) / mad

    # Correlation criterion
    samples_per_window = srate * window_width
    n_windows = int(x.shape[1] / samples_per_window)
    windows_allowed_below = n_windows * perc_of_windows
    max_corr_vals = np.ones((x.shape[0], n_windows))
    for i in range(2):
        curr_start = int(i * samples_per_window)
        curr_stop = int(curr_start + samples_per_window)
        curr_corr = np.corrcoef(x[:, curr_start:curr_stop])
        mask = ~np.eye(curr_corr.shape[0], dtype=bool)
        curr_corr = curr_corr * mask  # set diagonal els to 0
        max_corr_vals[:,i] = np.max(abs(curr_corr), axis=0)
    windows_below_threshold = np.sum(max_corr_vals<.5, axis=1)

    bads_idx = (np.where((robust_z > 5) |
                         (windows_below_threshold > windows_allowed_below)))

    return bads_idx


def notch_filter(data, target_freq, srate, quality_factor=30):
    """Applies IIR notch filter.

    Params
    ------
        data : ndArray
            signal to be filtered
        target_freq : int
            frequency to be filtered out
        srate : float
            sampling rate of the signal
        quality_factor: int

    Returns
    -------
        filtered : ndArray
            notch-filtered signal
    """
    x = butter_highpass(data, .3, srate)
    nyq = 0.5 * srate
    w0 = target_freq / nyq
    b, a = iirnotch(w0=w0, Q=quality_factor)
    return lfilter(b, a, x)


def butter_lowpass(data, h_cutoff, srate, order=5):
    """Applies low-pass filter.

    Params
    ------
        data : ndArray
            signal to be filtered
        h_cutoff : int
            cutoff above which to filter out all frequencies
        srate : float
            sampling rate of the signal
        order : int
            order of the filter

    Returns
    -------
        filtered : ndArray
            lowpass-filtered signal
    """
    nyq = 0.5 * srate
    normal_cutoff = h_cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if isinstance(data, Data):
        _data = lfilter(b, a, data.data, axis=-2)
        data = data.copy(data=_data)
    else:
        data = lfilter(b, a, data)
    return data


def butter_highpass(data, l_cutoff, srate, order=5):
    """ Applies high-pass filter.

    Params
    ------
        data : ndArray
            signal to be filtered
        l_cutoff : int
            cutoff below which to filter out all frequencies
        srate : float
            sampling rate of the signal
        order : int
            order of the filter

    Returns
    -------
        filtered : ndArray
            highpass-filtered signal
    """
    nyq = 0.5 * srate
    normal_cutoff = l_cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if isinstance(data, Data):
        _data = lfilter(b, a, data.data, axis=-2)
        data = data.copy(data=_data)
    else:
        data = lfilter(b, a, data)
    return data


def window_bandpass(data, l_cutoff, h_cutoff, srate, numtaps=400):
    """"""
    filt = signal.firwin(numtaps, [l_cutoff/srate, h_cutoff/srate], pass_zero=False)
    if isinstance(data, Data):
        filt = filt.reshape(-1, 1)
        _data = signal.convolve(data.data, filt, mode='same')
        data = data.copy(data=_data)
    else:
        data = signal.convolve(data, filt, mode='same')
    return data


def filter_raw(raw, srate, h_pass=False, l_pass=False, b_pass=False, l_cutoff=1,
               h_cutoff=100, order=5, numtaps=400):
    """Wrapper function for `butter_highpass`, `butter_lowpass`, `window_bandpass`

    Params
    ------
        raw : ndArray
            signal to be filtered
        h_pass : boolean
            whether high-pass filter should be applied
        l_pass : boolean
            whether low-pass filter should be applied
        l_cutoff : int
            cutoff below which to filter out all frequencies
        h_cutoff : int
            cutoff above which to filter out all frequencies
        srate : float
            sampling rate of the signal
        order : int
            order of the filter

    Returns
    -------
        filtered : ndArray
            lowpass-filtered signal

    See also
    --------
        :func: butter_highpass
        :func: butter_lowpass
    """
    filt = raw.copy()
    if not h_pass and not l_pass and not b_pass:
        msg = 'Have to choose either one of h_pass, l_pass, b_pass'
        raise ValueError(msg)
    if h_pass:
        filt = butter_highpass(data=filt, l_cutoff=l_cutoff, srate=srate,
                               order=order)
    if l_pass:
        filt = butter_lowpass(data=filt, h_cutoff=h_cutoff, srate=srate,
                              order=order)
    if b_pass:
        filt = window_bandpass(data=filt, l_cutoff=l_cutoff, h_cutoff=h_cutoff,
                               srate=srate, numtaps=numtaps)
    return filt


def append_subj_id(data, subj_id):
    """Appends subj_id to epoched `Data` object"""
    if not isinstance(data, Data):
        msg = 'Only wyrm.types.Data objects are supported.'
        raise TypeError(msg)
    n_epochs = data.data.shape[0]
    return np.repeat(subj_id, n_epochs)


def create_markers(data, win_length):
    """Creates markers of arbitrary name to later segment the data

    Params
    ------
        data : wyrm.Data
            data object to create markers for
        win_length : int
            length of each epoch, later used for segmenting

    Returns
    -------
        markers : list
            list of lists, each holding timepoint and arbitrary name
    """
    timepoints = data.axes[0]
    start = timepoints[0]
    stop = timepoints[-1]
    step = win_length*1000
    markers = []
    for idx in np.arange(start, stop, step):
        markers.append([idx, 'M1'])
    return markers


def fetch_labels(path_labels, study, subj_id):
    """Fetches label for a given subject

    Params
    ------
        path_labels : str
            path to CSV with label information
        study : str
            study to fetch labels for (see `constants` module for naming
            conventions)
        subj_id = list or str or int
            subj_id(s) to fetch labels for

    Returns
    -------
        labels : str or list of str
            label as string for one, list of strings for multiple subj

    See also
    --------
        :module: prepod.lib.constants
    """
    if not isinstance(subj_id, list):
        subj_id = [subj_id]
    subj_ids = [str(el) for el in subj_id]
    with open(path_labels, 'r') as f:
        dat = pd.read_csv(f, dtype='object')
    name_subj_id = const.CSV_COLNAMES[study]['subj_id']
    name_label = const.CSV_COLNAMES[study]['label']
    labels_to_drop = [el for el in list(dat) if el not in [name_subj_id, name_label]]
    dat = dat.drop(labels=labels_to_drop, axis=1)
    dat = dat[dat[name_subj_id].isin(subj_ids)]
    if dat.shape[0] == 1:
        return str(dat[name_label].iloc[0])
    else:
        return list(dat[name_label])


def merge_subjects(l, path_out=None):
    """Merges list of `Data` objects to one `Data` object

    Mainly delegates the call to wyrm.proccessing.append

    Params
    ------
        l : list
            list of `Data` objects to merge
        path_out : str
            if not None, path to store merged data

    Returns
    -------
        data : `Data` object
            merged data

    See also
    --------
        :type: wyrm.types.Data
        :func: wyrm.processing.append
    """
    if not isinstance(l, list):
        msg = 'Please provide list of `Data` objects to merge, got {}'.format(
            str(type(l))
        )
        raise TypeError(msg)
    if not all(isinstance(el, Data) for el in l):
        msg = 'All objects in list must be of type `wyrm.types.Data`'
        raise TypeError(msg)
    if len(l) < 2:
        msg = 'At least two objects are needed for merge.'
        raise TypeError(msg)

    data, new_file_needed, file_counter = None, False, 0
    for idx, el in enumerate(l):
        if idx == 0 or new_file_needed:
            data = el
            if new_file_needed:
                new_file_needed = False
        else:
            data = wyrm_append(data, el, extra=['bis', 'subj_id', 'markers'])

        print('Merged set {}/{}'.format(idx+1, len(l)))
        size = sys.getsizeof(data.data)
        print('Size in MB: {:.3f}'.format(size/1e6))

        # Write to file on last iteration or if file size would
        # potentially grow to beyond 4 GB
        if idx+1 == len(l) or size/1e6 > 3500:
            if path_out:
                file_counter += 1
                old_fname = path_out.split('/')[-1].split('.')[-2]
                new_fname = '{}_0{}'.format(old_fname, file_counter)
                new_path_out = path_out.replace(old_fname, new_fname)
                try:
                    io.save_as_pickled(data=data, path_out=new_path_out)
                except Exception:
                    print('Unable to save data.')
                finally:
                    if idx+1 != len(l):
                        new_file_needed = True

    return data


def match_bis(data, path_bis):
    """Matches each sample in `data` with a corresponding BIS value

    Will match at start and end of each recording, thus drops samples
    before both EEG or BIS have started and after one of them has
    finished.

    Params
    ------
        data : wyrm.types.Data
            data to merge BIS with
        path_bis : str
            path to BIS dir

    Returns
    -------
        data : wyrm.types.Data
            updated data object with attribute data.bis
    """
    # TODO: Align on ms instead of s? Align on every BIS value (Time stamp)
    # TODO: Drop where time jump in BIS >> 1s?
    # TODO: Find 'drop post-BIS EEG' bug
    bis = io.read_bis(path_bis)
    fs_eeg = data.fs
    eeg_start = data.starttime
    bis_start = bis['SystemTime'].iloc[0]
    bis_end = bis['SystemTime'].iloc[bis.shape[0]-1]
    bis['t_delta'] = bis['SystemTime'].diff().apply(lambda x: x.total_seconds())

    # drop samples pre-BIS recording
    time_delta = bis_start-eeg_start
    time_delta_s = time_delta.days * 24 * 3600 + time_delta.seconds

    if time_delta_s < 0:  # drop bis vals if start before eeg start
        bis = bis.drop(bis[bis['SystemTime'] < eeg_start].index)
        bis.reset_index(inplace=True)
        bis_start = bis['SystemTime'].iloc[0]
        time_delta = bis_start - eeg_start
        time_delta_s = time_delta.days * 24 * 3600 + time_delta.seconds

    new_start = int(time_delta_s * fs_eeg)
    data.data = data.data[new_start:]
    data.axes[0] = data.axes[0][new_start:]

    # drop samples post-BIS recording
    bis_values = np.array(bis['BIS'])
    time_delta = bis_end - bis_start
    time_delta_s = time_delta.days * 24 * 3600 + time_delta.seconds
    n_samples = int(time_delta_s * fs_eeg)
    data.data = data.data[:n_samples]
    data.axes[0] = data.axes[0][:n_samples]

    # create array of BIS vals, one per sample in the EEG
    t_deltas = np.append(np.delete(np.array(bis['t_delta']), 0), 0)
    n_repeats = np.nan_to_num(t_deltas * fs_eeg, copy=True).astype('int')
    bis_values = np.repeat(bis_values, n_repeats)  # one bis val per sample in eeg
    bis_values = bis_values[:data.data.shape[0]]  # drop post-EEG BIS
    data.data = data.data[:bis_values.shape[0]]  # drop post-BIS EEG
    data.axes[0] = data.axes[0][:bis_values.shape[0]]
    data.bis = bis_values

    print('Successfully aligned {} with BIS values.'.format(
        path_bis.split('/')[-2]
    ))

    return data


def segment_data(data, win_length):
    """Splits continuous signal into windows of variable length

    Mainly delegates the call to `wyrm.processing.segment_dat`.

    Params
    ------
        data : `Data`
            continuous signal
        win_length : int
            length of window in seconds

    Returns
    -------
        data : `Data`
            chunked data

    See also
    --------
        :func: wyrm.processing.segment_dat
    """
    label = data.label
    if not isinstance(label, str):
        label = str(label)
    marker_def = {label: ['M1']}
    ival = [0, win_length*1000]
    data = segment_dat(dat=data, marker_def=marker_def, ival=ival, timeaxis=0)
    data.axes[0] = np.repeat(label, data.axes[0].shape[0])  # update class names

    # segment_dat drops samples at the borders. Since markers are defined
    # starting at t0, it will always only drop samples at the end (if at all).
    # Drop associated BIS samples before reshaping.
    data.bis = data.bis[:data.data.shape[0]*data.data.shape[1]]
    data.bis = data.bis.reshape([data.data.shape[0], -1])

    # drop nan epochs
    # this is handled here, not earlier, as to not drop samples here and there,
    # but instead exclude whole windows if at least one sample is missing, thus
    # keeping meaningful distance between any two time points
    mask1 = np.all(~np.any(np.isnan(data.data), axis=1), axis=1)
    mask2 = ~np.any(np.isnan(data.bis), axis=1)
    mask = mask1 & mask2
    data.data = data.data[mask]
    data.axes[0] = data.axes[0][mask]
    data.bis = data.bis[mask]

    return data


def subset_data(data, bis_crit=None, drop_perc=None, drop_from='beginning', subj_ids=None):
    """Subsets an epoched Data object by BIS value and intra-OP time

    It might be useful to only look at data aligned with critical BIS
    values (e. g., BIS < 60) or only the second half of the OP.

    Params
    ------
        data : wyrm.types.Data
            epoched data to subset
        bis_crit : int
            critical BIS value, drop all epochs with BIS > `bis_crit`
        drop_perc : float
            percentage of OP to be dropped
        drop_from : str
            if drop_perc is passed, will drop from beginning or end of OP
            * 'beginning': drop from beginning
            * 'end': drop from end

    Returns
    -------
        data : wyrm.types.Data
            subsetted data
    """
    # TODO: Handle drop_perc == None
    if subj_ids:
        if (not isinstance(subj_ids, np.ndarray)
                and not isinstance(subj_ids, list)):
            msg = 'When passing `subj_ids`, have to pass list or np.ndarray.'
            raise TypeError(msg)

    dat = data.data.copy()
    bis = data.bis.copy()

    axes = data.axes.copy()
    subj_id = data.subj_id.copy()

    # only keep windows where BIS <= bis_crit
    if bis_crit:
        new_idx = np.where(np.all(bis <= bis_crit, axis=1))
        dat = dat[new_idx]
        bis = bis[new_idx]
        axes[0] = axes[0][new_idx]
        subj_id = subj_id[new_idx]

    data.data = dat
    data.bis = bis
    data.axes = axes
    data.subj_id = subj_id

    dat = data.data.copy()
    subj_id = data.subj_id.copy()

    if drop_perc:
        if drop_from not in ['beginning', 'end']:
            msg = 'drop_from must be one of \'beginning\', \'end\'.'
            raise ValueError(msg)

        unique_subj_ids = np.unique(subj_id)
        idx_to_keep = []
        for subj in unique_subj_ids:
            idx_subj = np.where(subj_id == subj)[0]
            data_subset = dat[idx_subj, :, :].squeeze()
            n_epochs = data_subset.shape[0]
            n_samples_per_epoch = data_subset.shape[1]
            n_samples = n_epochs * n_samples_per_epoch
            n_samples_to_drop = drop_perc * n_samples
            n_epochs_to_drop = int(np.floor(n_samples_to_drop/n_samples_per_epoch))
            if drop_from == 'beginning':
                idx_to_keep.append(idx_subj[n_epochs_to_drop:])
            else:
                idx_to_keep.append(idx_subj[:-n_epochs_to_drop])

        idx_to_keep = np.concatenate(idx_to_keep).ravel()
        data = select_epochs(data, indices=idx_to_keep)
        data.subj_id = data.subj_id[idx_to_keep]

    if subj_ids:
        mask = np.isin(data.subj_id, subj_ids)
        data.subj_id = data.subj_id[mask]
        dat = data.data.compress(mask, 0)  # classaxis is 0
        axes = data.axes[:]
        axes[0] = data.axes[0].compress(mask)
        data = data.copy(data=dat, axes=axes)

    return data


def create_fvs(data):
    """Creates 2D feature vectors from 3D Data object

    Epoched data is stored as 3D obj in wyrm (labels, samples, channels),
    most machine learning toolboxes (incl. wyrm), however, need 2D
    representations of the data (labels, samples * channels).

    Params
    ------
        data : wyrm.Data
            Data object with 3D axes

    Returns
    -------
        data : wyrm.Data
            Data object with 2D axes

    See also
    --------
        :type: wyrm.Data
    """
    dat = data.data.reshape((data.axes[0].shape[0], -1))
    axes = data.axes[:2]
    axes[-1] = np.arange(data.data.shape[-1])
    names = data.names[:2]
    names[-1] = 'feature_vector'
    units = data.units[:2]
    units[-1] = 'dl'

    return data.copy(data=dat, axes=axes, names=names, units=units)


def apply_csp(data, return_as='filtered', time_axis=1):
    """Calculates and applies CSP"""
    w, a, d = calculate_csp(data)
    filtered = apply_spatial_filter(data, w)
    if return_as == 'logvar':
        filtered.data = np.log(np.var(filtered.data, axis=time_axis))
        filtered.axes[1] = np.arange(filtered.data.shape[0])
        return filtered
    else:
        return filtered


def drop_na(data, cols):
    """Drops rows with have at least one NaN value from df"""
    for col in cols:
        data = data[pd.notnull(data[col])]
    return data


def drop_if_too_many_nans(data, threshold=.33):
    """Drops column from df if it has too many NaN values"""
    n_subjects = data.shape[0]
    crit_nans_value = np.floor(n_subjects*threshold).astype(int)
    keep = np.array(list(data))[(data.isna().sum() < crit_nans_value).values]
    return data[keep]


def drop_non_feature_cols(data, target):
    """"""
    keep = [el for el in list(data)
            if 'delir' not in el and 'op' not in el  and 'has' not in el]
    keep += [target]
    return data[keep]


def to_fv(data, feature_labels, target_labels):
    """Takes df as input and returns dict of ndarrays"""
    X = np.array(data[feature_labels])
    y = np.array(data[target_labels])
    data = {
        'X': X,
        'y': y,
        'X_labels': np.array(feature_labels),
        'y_labels': np.array(target_labels)
    }
    return data

