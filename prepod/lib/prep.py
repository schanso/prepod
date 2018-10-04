import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from wyrm.types import Data


def feature_vector(data, names=('class', 'amplitude'), units=('#', 'µV')):
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
    n_subj = data.data.shape[0]
    n_samples = data.data.shape[1]
    n_chans = data.data.shape[2]
    dat = data.data.reshape((n_subj, n_samples*n_chans))
    ax = [data.axes[0], np.linspace(0, 1, len(data.axes[1])*n_chans)]
    return Data(data=dat, axes=ax, names=names, units=units)


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
    return lfilter(b, a, data)


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
    return lfilter(b, a, data)


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

