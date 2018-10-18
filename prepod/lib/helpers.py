import os

from prepod.lib.constants import KNOWN_ERRORS, FORMATS


def fix_known_errors(d):
    """"""
    if d['subj_id'] in KNOWN_ERRORS.keys():
        exec(KNOWN_ERRORS[d['subj_id']])
    return d


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
        l : list or str
            file name(s) in `dir_in` (that optionally contain substr)
    """
    l = [f for f in os.listdir(dir_in)
         if (os.path.isfile(os.path.join(dir_in, f)) and not f.startswith('.'))]
    if substr:
        l = [f for f in l if substr in f]
    if sort_list:
        l.sort()
    if len(l) == 1:
        l = l[0]
    return l

