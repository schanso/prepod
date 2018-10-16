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
    'delta': (.1, 3.99),
    'theta': (4, 7.99),
    'alpha': (8, 12.99),
    'beta': (13, 29.99),
    'gamma': (30, 64)  # upper limit 64 because srate is 128
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

