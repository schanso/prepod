# Support
SUPPORTED_FTYPES = ['edf', 'eeg']
SUPPORTED_FORMATS = ['wyrm', 'dict']
SUPPORTED_REGIONS = ['central', 'frontal', 'parietal', 'occipital', 'temporal']

# Formats
FORMATS = {
    'datetime': '%Y-%m-%d %H:%M:%S.%f'
}

# Errors
KNOWN_ERRORS = {
    '2300': ('new_date = datetime.datetime.strftime(d[\'starttime\'], \'%Y-%d-%m %H:%M:%S.%f\');'
             + 'd[\'starttime\'] = datetime.datetime.strptime(new_date, FORMATS[\'datetime\'])')
}

# Sudocu
COLNAME_SUBJID_SUDOCU = 'case_no'
COLNAME_TARGET_SUDOCU = 'Nudesc_AWR_60min_Delir_ohne_Missings'

# Biocog
COLNAME_SUBJID_BIOCOG = 'BIC_ID'
COLNAME_TARGET_BIOCOG = 'POD_Ja_Nein'

