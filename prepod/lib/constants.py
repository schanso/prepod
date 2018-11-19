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

SUBJ_INFO_COLNAMES = {
    'Sudocu': [
        {'old_label': 'case_no', 'new_label': 'subj_id', 'dtype': 'object'},
        {'old_label': 'Geschlecht', 'new_label': 'gender', 'dtype': 'category'},
        {'old_label': 'Age_in_years', 'new_label': 'age', 'dtype': 'int8'},
        {'old_label': 'Nudesc_AWR_60min_Delir_ohne_Missings', 'new_label': 'delir_60min', 'dtype': 'category'},
        {'old_label': 'Nudesc_OP_Tag_Delir', 'new_label': 'delir_day1', 'dtype': 'category'},
        {'old_label': 'aufPropofol', 'new_label': 'med_auf_propofol', 'dtype': 'category'},
        {'old_label': 'aufIsoflurane', 'new_label': 'med_auf_isoflurane', 'dtype': 'category'},
        {'old_label': 'aufDesflurane', 'new_label': 'med_auf_desflurane', 'dtype': 'category'},
        {'old_label': 'aufSevoflurane', 'new_label': 'med_auf_sevoflurane', 'dtype': 'category'},
        {'old_label': 'benzodiazepine', 'new_label': 'med_benzodiazepine', 'dtype': 'category'},
        {'old_label': 'OP_Datum_A', 'new_label': 'op_date', 'dtype': 'datetime64'},
        {'old_label': 'EEG_Durchgefuehrt', 'new_label': 'has_eeg', 'dtype': 'category'},
        {'old_label': 'Summe_Kristalloide', 'new_label': 'sum_crystalloid', 'dtype': 'float'},
        {'old_label': 'Summe_Kolloide', 'new_label': 'sum_colloid', 'dtype': 'float'},
        {'old_label': 'Glucose_max', 'new_label': 'glucose_max', 'dtype': 'float'},
        {'old_label': 'Glucose_min', 'new_label': 'glucose_min', 'dtype': 'float'},
        {'old_label': 'Hb_max', 'new_label': 'hb_max', 'dtype': 'float'},
        {'old_label': 'Hb_min', 'new_label': 'hb_min', 'dtype': 'float'},
        {'old_label': 'Temperatur_max', 'new_label': 'temp_max', 'dtype': 'float'},
        {'old_label': 'Temperatur_min', 'new_label': 'temp_min', 'dtype': 'float'},
        {'old_label': 'BIS_angewendet', 'new_label': 'has_bis', 'dtype': 'category'},
        {'old_label': 'S_Beginn', 'new_label': 'op_starttime', 'dtype': 'object'},
        {'old_label': 'S_Ende', 'new_label': 'op_endtime', 'dtype': 'object'},
        {'old_label': 'eindexamethason', 'new_label': 'med_ein_dexamethason', 'dtype': 'category'},
        {'old_label': 'einsetrone', 'new_label': 'med_ein_setrone', 'dtype': 'category'},
        {'old_label': 'eindimenhydrinat', 'new_label': 'med_ein_dimenhydrinat', 'dtype': 'category'},
        {'old_label': 'eindroperisol', 'new_label': 'med_ein_droperisol', 'dtype': 'category'},
        {'old_label': 'einmcp', 'new_label': 'med_ein_mcp', 'dtype': 'category'},
        {'old_label': 'exdexamethason', 'new_label': 'med_ex_dexamethason', 'dtype': 'category'},
        {'old_label': 'exsetrone', 'new_label': 'med_ex_setrone', 'dtype': 'category'},
        {'old_label': 'exdimenhydrinat', 'new_label': 'med_ex_dimenhydrinat', 'dtype': 'category'},
        {'old_label': 'exdroperisol', 'new_label': 'med_ex_droperisol', 'dtype': 'category'},
        {'old_label': 'exmcp', 'new_label': 'med_ex_mcp', 'dtype': 'category'},
        {'old_label': 'ausdexamethason', 'new_label': 'med_aus_dexamethason', 'dtype': 'category'},
        {'old_label': 'aussetrone', 'new_label': 'med_aus_setrone', 'dtype': 'category'},
        {'old_label': 'ausdimenhydrinat', 'new_label': 'med_aus_dimenhydrinat', 'dtype': 'category'},
        {'old_label': 'ausdroperisol', 'new_label': 'med_aus_droperisol', 'dtype': 'category'},
        {'old_label': 'ausmcp', 'new_label': 'med_aus_mcp', 'dtype': 'category'},
        {'old_label': 'einlPropofol', 'new_label': 'med_einl_propofol', 'dtype': 'category'},
        {'old_label': 'einlThiopental', 'new_label': 'med_einl_thiopental', 'dtype': 'category'},
        {'old_label': 'einlEtomidate', 'new_label': 'med_einl_etomidate', 'dtype': 'category'},
        {'old_label': 'einlIsoflurane', 'new_label': 'med_einl_isoflurance', 'dtype': 'category'},
        {'old_label': 'einlSevoflurane', 'new_label': 'med_einl_sevoflurane', 'dtype': 'category'},
        {'old_label': 'einlDesflurane', 'new_label': 'med_einl_desflurane', 'dtype': 'category'},
        {'old_label': 'einlNarkotikaAndere', 'new_label': 'med_einl_narcotics_other', 'dtype': 'category'},
        {'old_label': 'einlFentanyl', 'new_label': 'med_einl_fentanyl', 'dtype': 'category'},
        {'old_label': 'einlRemifentanyl', 'new_label': 'med_einl_remifentanyl', 'dtype': 'category'},
        {'old_label': 'einlAlfentanil', 'new_label': 'med_einl_alfentanil', 'dtype': 'category'},
        {'old_label': 'einlSufentanyl', 'new_label': 'med_einl_sufentanyl', 'dtype': 'category'},
        {'old_label': 'einlOpiateAndere', 'new_label': 'med_einl_opioids_other', 'dtype': 'category'},
        {'old_label': 'aufAndere', 'new_label': 'med_auf_others', 'dtype': 'category'},
        {'old_label': 'aufRemifentanyl', 'new_label': 'med_auf_remifentanyl', 'dtype': 'category'},
        {'old_label': 'aufFentanyl', 'new_label': 'med_auf_fentanyl', 'dtype': 'category'},
        {'old_label': 'aufAlfentanil', 'new_label': 'med_auf_alfentanil', 'dtype': 'category'},
        {'old_label': 'aufSufentanyl', 'new_label': 'med_auf_sufentanyl', 'dtype': 'category'},
        {'old_label': 'flumazenil', 'new_label': 'med_flumanzenil', 'dtype': 'category'},
        {'old_label': 'naloxon', 'new_label': 'med_naloxon', 'dtype': 'category'},
        {'old_label': 'saggamadex', 'new_label': 'med_saggamadex', 'dtype': 'category'},
        {'old_label': 'neostigmin', 'new_label': 'med_neostigmin', 'dtype': 'category'}
    ],
    'Biocog': [

    ]
}

