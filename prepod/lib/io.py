import logging
import os
import re

import mne
import numpy as np
import pandas as pd
from wyrm.types import Data

from prepod.lib.globals import (SUPPORTED_FTYPES, SUPPORTED_FORMATS,
                                SUPPORTED_REGIONS)


logging.getLogger('mne').setLevel(logging.ERROR)


def strip_ch_names(raw):
    """Strips noise from `mne.Raw.ch_names`"""
    regex = r'(EEG |-\(A1\+A2\)\/)'
    old_names = raw.ch_names
    new_names = [re.sub(regex, '', el) for el in old_names]
    new_names = dict(zip(old_names, new_names))
    raw.rename_channels(new_names)
    return raw


def return_fnames(dir_in, substr=None, sort_list=True):
    """Returns list of file names in `dir_in`

    Params
    ------
        substr : str or None
            if not None, returns subset of file names that include substr
        sort_list : boolean
            if True, sorts list before returning

    Returns
    -------
        l : list
            file names in `dir_in` (that optionally contain substr)
    """
    l = [f for f in os.listdir(dir_in)
         if (os.path.isfile(os.path.join(dir_in, f)) and not f.startswith('.'))]
    if substr:
        l = [f for f in l if substr in f]
    if sort_list:
        l.sort()
    return l


def read_raw(path_in, region='frontal', ftype='eeg', drop_ref=True,
             drop_stim=True, n_samples=None, return_type='wyrm'):
    """Converts raw EEG into `dict`/`wyrm.Raw` using `mne.Raw`

    Params
    ------
        path_in : str
            path to raw file
        region : str or None
            regions to return data for; supported strings include:
            'central', 'frontal', 'parietal', 'occipital', 'temporal';
            if None, all channels will be returned
        ftype: str
            file type of raw file, supported formats are read from
            globals.`SUPPORTED_FTYPES`
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
                          'markers': list)

    Raises
    ------
        TypeError if `ftype`, `region`, or `return_type` not supported

    See also
    --------
        :type: wyrm.Data
    """
    path_in = path_in.strip()
    if ftype in SUPPORTED_FTYPES:
        try:
            if ftype == 'edf':
                raw = mne.io.read_raw_edf(path_in, preload=True)
            elif ftype == 'eeg':
                path_in = path_in.replace(ftype, 'vhdr')
                raw = mne.io.read_raw_brainvision(path_in, preload=True)
            else:
                msg = 'File type {} not supported.'.format(ftype)
                raise TypeError(msg)

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

            d = {
                'signal': signal,
                'srate': raw.info['sfreq'],
                'ch_names': raw.info['ch_names'],
                'n_chans': len(raw.info['ch_names']),
                'time_points': times,
                'markers': []
            }
            print('Successfully read file ' + path_in)

            if return_type == 'wyrm':
                data = d['signal'].transpose()
                axes = [d['time_points'], d['ch_names']]
                names = ['time', 'channels']
                units = ['s', 'µV']
                data = Data(data, axes, names, units)
                data.fs = d['srate']
                data.markers = d['markers']
                return data
            elif return_type == 'dict':
                return d
            else:
                msg = 'Return_type {} not supported.'.format(return_type)
                raise TypeError(msg)

        except Exception as e:
            print('Failed to load file ' + path_in + '.\n\n Raised: ' + str(e))

    else:
        msg = ('File format \'{}\' not supported. '
               + 'Use one of {}.').format(ftype, str(SUPPORTED_FTYPES))
        raise TypeError(msg)


def store_raws(dir_in, dir_out, ftype_in, out_format='wyrm', subset=None):
    """Wrapper function for `read_raw`, converts raw files in dir to .npy

    Params
    ------
        dir_in : str
            path to dir with raw files
        dir_out : str
            path to dir to write to
        ftype_in : str
            file type of raw file
        out_format : str
            whether data should be stored wyrm- or np-ready
        subset : str
            if not None, indicating channel subset by region

    Returns
    -------
        None

    Raises
    ------
        TypeError if `ftype_in` or `out_format` not supported

    See also
    --------
        :func: read_raw
        :global: SUPPORTED_FTYPES
        :global: SUPPORTED_FORMATS
        :global: SUPPORTED_REGIONS
    """
    if ftype_in not in SUPPORTED_FTYPES:
        msg = ('File type \'{}\' currently not supported. '
               + 'Choose one of {}.').format(ftype_in, str(SUPPORTED_FTYPES))
        raise TypeError(msg)

    if out_format not in SUPPORTED_FORMATS:
        msg = ('Format \'{}\' not supported. '
               + 'Choose one of {}.').format(out_format, str(SUPPORTED_FORMATS))
        raise TypeError(msg)

    ext = ftype_in
    if ftype_in == 'eeg':
        ext = 'vhdr'

    fnames = return_fnames(dir_in, substr=ext, sort_list=True)
    for fname_in in fnames:
        path_in = dir_in + fname_in
        path_out = dir_out + fname_in.replace(ext, 'npy')
        try:
            data = read_raw(path_in=path_in, ftype=ftype_in,
                            return_type=out_format, region=subset)
            np.save(path_out, data)
            print('Successfully wrote data to ' + path_out)
        except Exception as e:
            msg = ('Failed to write data to {}.'
                   + '\n\nRaised: {}').format(path_out, str(e))
            print(msg)

        # TODO: Implement concatenation
        # if concat:
        #     print('Concatinating...')
        #     fnames = return_fnames(dir_out)
        #     subj_ids = sorted(list(set([el.split('_')[0] for el in fnames])))
        #     for subj_id in subj_ids:
        #         fpaths = sorted([dir_out + fname for fname in fnames
        #                          if subj_id in fname])
        #         data = np.load(fpaths[0])
        #         if len(fpaths) > 1:
        #             for fpath in fpaths[1:]:
        #                 data = np.concatenate((data, np.load(fpath)), axis=1)
        #             print('Successfully merged ' + str(len(fpaths)) + ' files for '
        #                   + str(subj_id) + '.')
        #         np.save(file=fpaths[0].replace('01', 'complete'), arr=data)
        #         print('Successfully wrote data to '
        #               + fpaths[0].replace('01', 'complete'))


def import_folder(dir_in, substr=None, exclude=[], in_format='wyrm'):
    """Imports folder of .npy files and returns them as list

    Params
    ------
        dir_in : str
            path to dir with .npy files
        substr : str
            if not None, only file names including `substr` are loaded
        exclude : list of str
            indicating which (if any) files should be excluded
        in_format : str
            format to read from file

    Returns
    -------
        data : list
            list of ndArrays

    Raises
    ------
        ValueError if file to be excluded is not in dir.
    """
    if in_format not in SUPPORTED_FORMATS:
        msg = ('Format \'{}\' not supported. '
               + 'Use one of {}.').format(in_format, str(SUPPORTED_FORMATS))
        raise TypeError(msg)

    fnames = return_fnames(dir_in, substr=substr)
    if len(exclude):
        try:
            [fnames.remove(el) for el in exclude]
        except ValueError:
            msg = ('Seems at least one of the file names to be excluded does '
                   + 'not exist in {}.').format(dir_in)
            raise ValueError(msg)

    data = []
    fpaths = [dir_in + f for f in fnames]
    for fpath in fpaths:
        _data = np.load(fpath)
        if in_format == 'wyrm':
            _data = _data.flatten()[0]
        data.append(_data)
        print('Successfully read file ' + fpath)
    return data


def import_targets(fpath, colname_subjid, colname_target, subj_ids=None):
    """Imports targets from CSV table

    Params
    ------
        fpath : str
            String to CSV file
        colname_subjid : str
            column name of subject ids
        colname_target : str
            column name of targets
        subj_ids : list or None
            if not None, creates subset of targets for subj_ids

    Returns
    -------
        targets : ndArray
            array of target labels
    """
    df = pd.read_csv(fpath, dtype='str')
    targets = df[[colname_subjid, colname_target]]
    targets.columns = ['case_no', 'target']
    targets['target'] = targets['target'].astype('int')
    if subj_ids:
        targets = targets[targets['case_no'].isin(subj_ids)]
    return np.array(targets['target'])



def append_labels(data, labels):
    """Stores class labels in epoched `Data` objects

    wyrm stores epoched EEG data in one Data object, its `axes` attribute
    holding an array of class labels for each epoch. To merge `Data` obj
    with class names usually stored in CSV-like files, `append_labels`
    loops over each epoch in `Data` and create label info.

    Params
    ------
        data : wyrm.Data
            Data object holding epoched EEG data in data.data
        labels : list
            list of class labels, have to correspond to epochs by index

    Returns
    -------
        updated_data : list
            list of `wyrm.Data` objects, each holding on epoch

    See also
    --------
        :func: read_raw
        :type: wyrm.Data
    """
    updated_data = []
    for idx, dataset in enumerate(data):
        dataset.data = np.expand_dims(dataset.data, axis=0)
        dataset.names.insert(0, 'class')
        dataset.axes.insert(0, labels[idx])
        dataset.units.insert(0, '#')
        updated_data.append(dataset)
    return updated_data

