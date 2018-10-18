import datetime
import logging
import os
import pickle
import re
import sys

import mne
import numpy as np
import pandas as pd
from wyrm.types import Data

from prepod.lib.constants import (SUPPORTED_FTYPES, SUPPORTED_REGIONS, FORMATS)
from prepod.lib.helpers import fix_known_errors, return_fnames

logging.getLogger('mne').setLevel(logging.ERROR)


def strip_ch_names(raw):
    """Strips noise from `mne.Raw.ch_names`"""
    regex = r'(EEG |-\(A1\+A2\)\/)'
    old_names = raw.ch_names
    new_names = [re.sub(regex, '', el) for el in old_names]
    new_names = dict(zip(old_names, new_names))
    raw.rename_channels(new_names)
    return raw


def parse_raw(path_in, dir_out=None, ftype=None, region='frontal', drop_ref=True,
              drop_stim=True, n_samples=None, return_type='wyrm'):
    """Converts raw EEG into `dict`/`wyrm.Raw` using `mne.Raw`

    Params
    ------
        path_in : str
            path to raw file
        dir_out : str or None
            if not None, will store raw signals in dir_out (.npy)
        ftype: str
            file type of raw file, supported formats are read from
            prepod.lib.constants.`SUPPORTED_FTYPES`
        region : str or None
            regions to return data for; supported strings include:
            'central', 'frontal', 'parietal', 'occipital', 'temporal';
            if None, all channels will be returned
        drop_ref : boolean
            whether reference electrodes should be dropped
        drop_stim : boolean
            whether stimulus electrodes should be dropped
        n_samples : int
            return subset of n_samples from start (mainly development)
        return_type : str
            one of ['wyrm'|'dict'], indicating what format to return the
            data in ('wyrm' -> wyrm.Data, 'dict' -> dict of ndArrays)

    Returns
    -------
        raw : dict or wyrm.Data
            dict -> {'signal': ndArray [channels x time], 'srate': float,
                     'ch_names': list, 'n_chans_: int,
                     'time_points': ndArray, 'markers': list}
            wyrm.Data -> ('data': ndArray [time x channels], axes: list,
                          'names': list, 'units': list, 'fs': float,
                          'markers': list, 'starttime': datetime)

    Raises
    ------
        TypeError if `ftype`, `region`, or `return_type` not supported

    See also
    --------
        :type: wyrm.Data
    """
    if ftype not in SUPPORTED_FTYPES:
        msg = 'File type {} not supported. Choose one of {}'.format(
            ftype, ', '.join(SUPPORTED_FTYPES))
        raise TypeError(msg)

    if isinstance(path_in, list):
        paths = path_in
    else:
        if os.path.isdir(path_in):
            path_in = path_in + '/' if path_in[-1] != '/' else path_in
            paths = [path_in + el for el in return_fnames(path_in, substr=ftype)]
        else:
            paths = [path_in.strip()]

    subj_id = paths[0].split('/')[-1].split('_')[0]
    raw, raws = None, []
    for idx, path in enumerate(paths):
        if idx == 0:
            if ftype == 'edf':
                raw = mne.io.read_raw_edf(path, preload=True)
            elif ftype == 'eeg':
                path = path.replace(ftype, 'vhdr')
                raw = mne.io.read_raw_brainvision(path, preload=True)
            raw.cals = np.array([])
        else:
            if ftype == 'edf':
                _raw = mne.io.read_raw_edf(path, preload=True)
            elif ftype == 'eeg':
                path = path.replace(ftype, 'vhdr')
                _raw = mne.io.read_raw_brainvision(path, preload=True)
            raws.append(_raw)

    if len(raws):
        raw.append(raws)  # append multiple file to continuous signal

    raw = strip_ch_names(raw)
    if drop_ref:
        to_drop = [el for el in raw.ch_names if 'Ref' in el]
        raw.drop_channels(to_drop)
    if drop_stim:
        to_drop = [el for el in raw.ch_names if 'STI' in el]
        raw.drop_channels(to_drop)
    if region and region in SUPPORTED_REGIONS:
        to_drop = [el for el in raw.ch_names
                   if region[0].upper() not in el]
        raw.drop_channels(to_drop)
    else:
        print('Your region of interest is not supported. Choose one of '
              + str(SUPPORTED_REGIONS) + '. Will return full set.')

    if n_samples:
        signal = raw._data[:, :n_samples]
        times = raw.times[:n_samples]
    else:
        signal = raw._data
        times = raw.times

    start_time = datetime.datetime.utcfromtimestamp(
        raw.info['meas_date']).strftime(FORMATS['datetime'])

    d = {
        'signal': signal,
        'srate': raw.info['sfreq'],
        'ch_names': raw.info['ch_names'],
        'n_chans': len(raw.info['ch_names']),
        'time_points': times * 1000,  # convert s to ms
        'markers': [],
        'starttime': datetime.datetime.strptime(start_time, FORMATS['datetime']),
        'subj_id': subj_id
    }

    d = fix_known_errors(d)

    print('Successfully read file(s) ' + ', '.join(paths))

    if return_type == 'wyrm':
        data = d['signal'].transpose()
        axes = [d['time_points'], d['ch_names']]
        names = ['time', 'channels']
        units = ['s', 'µV']
        data = Data(data, axes, names, units)
        data.fs = d['srate']
        data.markers = d['markers']
        data.starttime = d['starttime']
    elif return_type == 'dict':
        data = d
    else:
        msg = 'Return_type {} not supported.'.format(return_type)
        raise TypeError(msg)

    if dir_out:
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        path_out = dir_out + paths[0].split('/')[-1].replace(ftype, 'npy')
        np.save(path_out, data)
        print('Successfully wrote data to ' + path_out)

    return data


def read_bis(path_in, from_type='bilateral'):
    """Reads in .asp files and returns BIS as a function of time

    Params
    ------
        path_in : str
            path to .asp file or folder containing multiple .asp files
        from_type : str
            which data to parse; Rugloop .asp files have three different
            types stored per file: '2channel', 'monitorrev', 'bilateral'

    Returns
    -------
        df : pd.DataFrame
            table with mean BIS values as a function of time
            (df.columns: ['RAppTime', 'SystemTime', 'BIS'])
    """
    is_dir = os.path.isdir(path_in)

    supported_ftypes = ['asp']
    if is_dir:
        if not path_in[-1] == '/':
            path_in = path_in + '/'
        fnames = return_fnames(dir_in=path_in, substr='asp')
        if isinstance(fnames, str):  # to list if only one file
            fnames = [fnames]
        if len(fnames) == 0:
            msg = 'No files of type `asp` in dir.'
            raise TypeError(msg)
        paths = [path_in + el for el in fnames]
    else:
        ftype = path_in.split('.')[-1]
        if ftype not in supported_ftypes:
            msg = 'File must be of type {}, got {}.'.format(
                str(supported_ftypes), ftype)
            raise TypeError(msg)
        paths = [path_in]

    supported_from_types = ['2channel', 'monitorrev', 'bilateral']
    idx_long, idx_short, idx_unit = (None, None, None)
    if from_type in supported_from_types:
        if from_type == '2channel':
            idx_long, idx_short, idx_unit = (0, 1, 2)
        elif from_type == 'monitorrev':
            idx_long, idx_short, idx_unit = (3, 4, 5)
        elif from_type == 'bilateral':
            idx_long, idx_short, idx_unit = (6, 7, 8)
    else:
        msg = '`from_type` must be one of {}, got {}.'.format(
            str(supported_from_types), str(from_type))
        raise TypeError(msg)

    df = pd.DataFrame(columns=['RAppTime', 'SystemTime', 'BIS'])
    for path in paths:
        data = open(path, encoding='ascii', errors='ignore')
        _df = pd.read_table(data, delimiter='|', dtype='object', skiprows=[0],
                            header=None)

        _df.columns = _df.iloc[idx_long, :]
        row_title = _df.iloc[idx_unit, 0]
        rows_to_drop = [el for el in range(9)]
        _df = _df.drop(_df.index[rows_to_drop])  # drop header rows
        _df = _df[_df.iloc[:, 0] == row_title]  # drop wrong-typed rows
        _df = _df.loc[:, _df.columns.notnull()]  # drop nan cols
        _df.reset_index(inplace=True, drop=True)

        alg_abbr = ['DB1', 'B3']
        bis_cols = [el for el in _df.columns for abbr in alg_abbr
                    if abbr in el  # keep only cols with used alg
                    and el[-1] in ['L', 'R']]  # drop duplicates (L==2, R==4)
        df_bis = _df[bis_cols].replace('xxx', np.nan).astype(float)
        df_bis = df_bis[df_bis.columns[(df_bis > 0).any()]]  # keep only used alg
        df_bis['BIS'] = df_bis.mean(axis=1)

        time_cols = ['RAppTime', 'SystemTime']
        df_time = _df[time_cols]

        _df = pd.concat([df_time, df_bis['BIS']], axis=1)  # keep mean only
        df = pd.concat([df, _df], axis=0)

    df['RAppTime'] = df['RAppTime'].astype('int64', copy=True)
    df['SystemTime'] = pd.to_datetime(df['SystemTime'], dayfirst=True)
    df.sort_values(by=['SystemTime'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def save_as_pickled(data, path_out):
    """"""
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(bytes_out)
    with open(path_out, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_wyrm(path):
    """Loads .npy files storing wyrm.Data objects"""
    data = np.load(file=path).flatten()[0]
    print('Successfully loaded data from {}.'.format(path))
    return data

