# Support
SUPPORTED_FTYPES = ['edf', 'eeg']
SUPPORTED_FORMATS = ['wyrm', 'dict']
SUPPORTED_REGIONS = ['central', 'frontal', 'parietal', 'occipital', 'temporal']

# Formats
FORMATS = {
    'datetime': '%Y-%m-%d %H:%M:%S.%f'
}

# Plot Styling
PLOT_STYLING = {
    'c': ['#f8766d', '#619cff', '#8da0cb', '#e78ac3', '#ffd92f'],
    'bc': '#f5f5f5',  # background color
    'gc': '#dcdcdc',  # grid line color
    'plot_styles': {'lw': 1, 'alpha': None}
}

# Errors
KNOWN_ERRORS = {}

# Exclude subjects
EXCLUDE_SUBJ = ['2183', '2482']

# Sudocu
COLNAME_SUBJID_SUDOCU = 'case_no'
COLNAME_TARGET_SUDOCU = 'Nudesc_AWR_60min_Delir_ohne_Missings'

# Biocog
COLNAME_SUBJID_BIOCOG = 'BIC_ID'
COLNAME_TARGET_BIOCOG = 'POD_Ja_Nein'

