# Support
SUPPORTED_FTYPES = ['edf', 'eeg']
SUPPORTED_FORMATS = ['wyrm', 'dict']
SUPPORTED_REGIONS = ['central', 'frontal', 'parietal', 'occipital', 'temporal']

# Formats
FORMATS = {
    'datetime': '%Y-%m-%d %H:%M:%S.%f'
}

# Frequency bands
FREQ_BANDS = {
    'total': (.1, 63.9),  # upper limit 64 because srate is 128
    'slow': (.1, 1.9),
    'delta': (.1, 3.9),
    'theta': (4, 7.9),
    'alpha': (8, 12.9),
    'beta': (13, 29.9),
    'gamma': (30, 63.9),
    'below20': (.1, 19.9),
    'below50': (.1, 49.9)
}

# Plot Styling
props_keys = ['xlim', 'ylim', 'xticklabels', 'yticklabels', 'xlabel', 'ylabel',
              'legend']
PLOT_STYLING = {
    'c': ['#f8766d', '#619cff', '#8da0cb', '#e78ac3', '#ffd92f'],
    'bc': '#f5f5f5',  # background color
    'gc': '#dcdcdc',  # grid line color
    'plot_styles': {'lw': 1, 'alpha': None},
    'props': {key: None for key in props_keys}
}

# Errors
KNOWN_ERRORS = {}

# Exclude subjects
EXCLUDE_SUBJ = ['2183', '2482']

# Sudocu
COLNAME_SUBJID_SUDOCU = 'case_no'
COLNAME_TARGET_SUDOCU = 'Nudesc_AWR_60min_Delir_ohne_Missings'  # 'Nudesc_OP_Tag_Delir'

# Biocog
COLNAME_SUBJID_BIOCOG = 'BIC_ID'
COLNAME_TARGET_BIOCOG = 'POD_Ja_Nein'

CSV_COLNAMES = {
    'Sudocu': {
        'subj_id': 'case_no',
        'label': 'Nudesc_AWR_60min_Delir_ohne_Missings'
    },
    'Biocog': {
        'subj_id': 'BIC_ID',
        'label': 'POD_Ja_Nein'
    }
}

