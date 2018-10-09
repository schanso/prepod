from prepod.lib.io import return_fnames, parse_raw


# PARSE RAW FILES AND STORE AS NPY

path_data = '/Users/jannes/Projects/delir/scripts/prepod/prepod/data/'
dir_raw = path_data + 'rec/sudocu/brainvision/raw/'
dir_out = dir_raw + 'npy/frontal/'
fnames = return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames])))

for subj_id in subj_ids:
    path = [dir_raw + el for el in fnames if subj_id in el]
    parse_raw(path_in=path, dir_out=dir_out, ftype='edf', region='frontal')