import pickle
import sys

import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from wyrm.processing import append as wyrm_append
from wyrm.types import Data

from prepod.lib.io import read_bis, save_as_pickled


def to_feature_vector(data, names=('class', 'amplitude'), units=('#', 'µV')):
    """Creates a 2D feature vector from 3D `wyrm.Data` object

    Epoched data is stored as 3D obj in wyrm (labels, samples, channels),
    most machine learning toolboxes (incl. wyrm), however, need 2D
    representations of the data (labels, (samples * channels)).

    Params
    ------
        data : wyrm.Data
            Data object with 3D axes
        names : iterable of str
            names for 2D data representation
        units : iterable of str
            units for 2D data representation

    Returns
    -------
        data : wyrm.Data
            Data object with 2D axes

    See also
    --------
        :type: wyrm.Data
    """
    if not isinstance(names, list):
        names = list(names)
    if not isinstance(units, list):
        units = list(units)

    n_epochs = data.data.shape[0]
    n_samples = data.data.shape[1]
    n_chans = data.data.shape[2]
    ax = [data.axes[0], np.linspace(0, 1, len(data.axes[1]) * n_chans)]
    fs = data.fs
    try:
        bis = data.bis
    except AttributeError:
        bis = None

    dat = data.data.reshape((n_epochs, n_samples * n_chans))
    data = Data(data=dat, axes=ax, names=names, units=units)
    data.fs = fs
    data.bis = bis

    return data


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


def filter_raw(raw, srate, h_pass=True, l_pass=True, l_cutoff=1, h_cutoff=100,
               order=5):
    """Wrapper function for `butter_highpass` and `butter_lowpass`

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
    if h_pass:
        filt = butter_highpass(data=filt, l_cutoff=l_cutoff, srate=srate,
                               order=order)
    if l_pass:
        filt = butter_lowpass(data=filt, h_cutoff=h_cutoff, srate=srate,
                              order=order)
    return filt


def align_bis(path_signal, path_bis):
    """"""
    # TODO: Align on ms instead of s? Align on every BIS value (Time stamp)
    # TODO: Drop where time jump in BIS >> 1s?
    # TODO: Find 'drop post-BIS EEG' bug
    if not isinstance(path_signal, str):
        msg = 'Please provide file path as str, got {}'.format(type(path_signal))
        raise TypeError(msg)

    data = np.load(file=path_signal).flatten()[0]
    bis = read_bis(path_bis)
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

    # drop samples post-BIS recording
    bis_values = np.array(bis['BIS'])
    time_delta = bis_end - bis_start
    time_delta_s = time_delta.days * 24 * 3600 + time_delta.seconds
    n_samples = int(time_delta_s * fs_eeg)
    data.data = data.data[:n_samples]
    eeg_dur = data.data.shape[0] / fs_eeg

    # create array of BIS vals, one per sample in the EEG
    t_deltas = np.append(np.delete(np.array(bis['t_delta']), 0), 0)
    n_repeats = np.nan_to_num(t_deltas * fs_eeg, copy=True).astype('int')
    bis_values = np.repeat(bis_values, n_repeats)  # one bis val per sample in eeg
    bis_values = bis_values[:data.data.shape[0]]  # drop post-EEG BIS
    data.data = data.data[:bis_values.shape[0]]  # drop post-BIS EEG
    data.fs = fs_eeg

    print('Successfully aligned {} with BIS values.'.format(
        path_signal.split('/')[-1]
    ))

    return data, bis_values


def split_into_wins(data, bis_values, win_length=5, bis_crit=None, keep_proportion=None):
    """Splits continuous signal into windows of variable length

    Params
    ------
        data : `Data`
            continuous signal
        bis_values : ndArray
            data-aligned BIS values
        win_length : int
            length of window in seconds
        bis_crit : int
            drop window if at least one sample is associated with a BIS
            value above critical level
        keep_proportion : float
            if not None, must be between 0 and 1; indicates the
            proportion of the data that should be kept, evaluated from
            the end, i. e. 0.33 -> keep only last third of the data,
            0.5 -> keep second half

    Returns
    -------
        data : `Data`
            chunked data
        bis_values : ndArray
            chuned BIS values

    See also
    --------
        :func: align_bis
    """
    fs_eeg = data.fs

    # keep only proportion of the data (counted from the end)
    if keep_proportion:
        n_samples_to_keep = int(np.floor(data.data.shape[0]*keep_proportion))
        data.data = data.data[-n_samples_to_keep:]
        bis_values = bis_values[-n_samples_to_keep:]

    # cut into windows of variable length
    win_samples = win_length * fs_eeg
    n_wins = np.floor(data.data.shape[0] / win_samples)
    new_start = int(data.data.shape[0] - (n_wins * win_samples))
    data.data = data.data[new_start:]
    bis_values = bis_values[new_start:]

    chunks_data = np.array(np.split(data.data, n_wins))
    chunks_bis = np.array(np.split(bis_values, n_wins))

    # only keep windows where BIS <= bis_crit
    if bis_crit:
        new_idx = np.where(np.all(chunks_bis <= bis_crit, axis=1))
        chunks_data = chunks_data[new_idx]
        chunks_bis = chunks_bis[new_idx]

    # to Data
    time_points = np.linspace(0, win_length, win_samples)
    axes = [np.arange(chunks_data.shape[0]), time_points, data.axes[1]]
    names = ['epoch', 'time', 'channels']
    units = ['#', 's', 'µV']
    data = Data(data=chunks_data, axes=axes, names=names, units=units)
    data.fs = fs_eeg
    data.bis = chunks_bis

    return data


def append_label(data, label):
    """Appends class label to epoched `Data` object"""
    if not isinstance(data, Data):
        msg = 'Only wyrm.types.Data objects are supported.'
        raise TypeError(msg)
    n_epochs = data.data.shape[0]
    data.axes[0] = np.repeat(label, n_epochs)

    return data


def append_subj_id(data, subj_id):
    """Appends subj_id to epoched `Data` object"""
    if not isinstance(data, Data):
        msg = 'Only wyrm.types.Data objects are supported.'
        raise TypeError(msg)
    n_epochs = data.data.shape[0]
    data.subj_id = np.repeat(subj_id, n_epochs)

    return data

def merge_subjects(l, path_out=None):
    """Merges list of `Data` objects to one `Data` object

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
        msg = 'At least two `Data` objects are needed to merge together.'
        raise TypeError(msg)


    n_channels = int(l[0].data.shape[1]/l[0].bis.shape[1])
    samples_per_epoch = l[0].bis.shape[1]

    bis = np.empty(shape=(0, samples_per_epoch), dtype='float64')
    subj_ids = np.empty(shape=(0,), dtype='<U4')
    for idx, el in enumerate(l):
        if idx == 0:
            data = el
            fs = data.fs
        else:
            data = wyrm_append(data, el)
        bis = np.concatenate((bis, el.bis))
        subj_ids = np.concatenate((subj_ids, el.subj_id))
        print('Appended {}/{}'.format(str(idx+1), str(len(l))))

    # Add fs and bis to merged Data object
    data.fs = fs
    data.bis = np.tile(bis, (1, n_channels))  # repeat bis for all five channels
    data.subj_ids = subj_ids

    if path_out:
        print('Saving...')
        np.save(path_out, arr=data)
        print('Successfully wrote merged data to ' + path_out)

    return data


def subset_data(data, bis_crit=None, drop_perc=None, drop_from='beginning'):
    """"""
    # TODO: Implement drop
    chunks_data = data.data
    chunks_bis = data.bis
    axes = data.axes
    subj_ids = data.subj_ids

    # only keep windows where BIS <= bis_crit
    if bis_crit:
        new_idx = np.where(np.all(chunks_bis <= bis_crit, axis=1))
        chunks_data = chunks_data[new_idx]
        chunks_bis = chunks_bis[new_idx]
        axes[0] = axes[0][new_idx]
        subj_ids = subj_ids[new_idx]

    # keep only proportion of the data (counted from drop_from)
    # if drop_perc:
    #     n_samples_to_drop = int(np.floor(chunks_data.shape[0] * drop_perc))
    #     if drop_from not in ['beginning', 'end']:
    #         msg = 'drop_from must be one of \'beginning\', \'end\'.'
    #         raise ValueError(msg)
    #     if drop_from == 'beginning':
    #         chunks_data = chunks_data[n_samples_to_drop:,:]
    #         chunks_bis = chunks_bis[n_samples_to_drop:,:]
    #         axes[0] = axes[0][n_samples_to_drop:]
    #         subj_ids = subj_ids[n_samples_to_drop:]
    #     else:
    #         chunks_data = chunks_data[:-n_samples_to_drop, :]
    #         chunks_bis = chunks_bis[:-n_samples_to_drop, :]
    #         axes[0] = axes[0][:-n_samples_to_drop]

    data.data = chunks_data
    data.axes = axes
    data.bis = chunks_bis
    data.subj_ids = subj_ids

    return data