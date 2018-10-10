import datetime

from prepod.lib.constants import KNOWN_ERRORS, FORMATS


def fix_known_errors(d):
    """"""
    if d['subj_id'] in KNOWN_ERRORS.keys():
        exec(KNOWN_ERRORS[d['subj_id']])
    return d

