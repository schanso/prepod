# Support
SUPPORTED_FTYPES = ['edf', 'eeg']
SUPPORTED_FORMATS = ['wyrm', 'dict']
SUPPORTED_REGIONS = ['full', 'central', 'frontal', 'parietal', 'occipital',
                     'temporal', 'fronto-parietal', 'pre-frontal']

# Formats
FORMATS = {
    'datetime': '%Y-%m-%d %H:%M:%S.%f'
}

# Frequency bands
FREQ_BANDS = {
    'total': (.5, 63),  # upper limit 64 because srate is 128
    'slow': (.5, 1.9),
    'delta': (.5, 3.9),
    'theta': (4, 7.9),
    'alpha': (8, 12.9),
    'beta': (13, 29.9),
    'gamma': (30, 63),
    'below10': (.5, 9.9),
    'below15': (.5, 14.9),
    'below20': (.5, 19.9),
    'below40': (.5, 39.9),
    'below50': (.5, 49)
}

# Sensors
CH_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
SENSOR_LOCS = [
    {'x': -0.139036, 'y': 0.427980, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Fp1'},
    {'x': 0.000000, 'y': 0.450000, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Fpz'},
    {'x': 0.139086, 'y': 0.427994, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Fp2'},
    {'x': -0.364052, 'y': 0.264534, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'F7'},
    {'x': -0.186808, 'y': 0.232745, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'F3'},
    {'x': 0.000000, 'y': 0.225021, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Fz'},
    {'x': 0.186808, 'y': 0.232745, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'F4'},
    {'x': 0.364052, 'y': 0.264534, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'F8'},
    {'x': -0.480000, 'y': 0.000046, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T3'},
    {'x': -0.364052, 'y': -0.264402, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T5'},
    {'x': -0.450000, 'y': 0.000014, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T7'},
    {'x': -0.225000, 'y': 0.000046, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'C3'},
    {'x': 0.000000, 'y': 0.000072, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Cz'},
    {'x': 0.225000, 'y': 0.000046, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'C4'},
    {'x': 0.480000, 'y': 0.000046, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T4'},
    {'x': 0.364052, 'y': -0.264402, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T6'},
    {'x': 0.450000, 'y': 0.000014, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'T8'},
    {'x': -0.364075, 'y': -0.264522, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'P7'},
    {'x': -0.186850, 'y': -0.232613, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'P3'},
    {'x': 0.000000, 'y': -0.224952, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Pz'},
    {'x': 0.186850, 'y': -0.232613, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'P4'},
    {'x': 0.364075, 'y': -0.264522, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'P8'},
    {'x': -0.139045, 'y': -0.427979, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'O1'},
    {'x': 0.000000, 'y': -0.450000, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'Oz'},
    {'x': 0.139045, 'y': -0.427979, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'O2'},
    {'x': -0.547000, 'y': -0.500000, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'COMNT'},
    {'x': 0.547000, 'y': -0.500000, 'width': 0.125128, 'height': 0.093849, 'ch_name': 'SCALE'}
]

# Plot Styling
props_keys = ['xlim', 'ylim', 'xticklabels', 'yticklabels', 'xlabel', 'ylabel',
              'legend', 'xticks', 'yticks']
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
        {'old_label': 'Geschlecht', 'new_label': 'gender', 'dtype': 'int64'},
        {'old_label': 'Age_in_years', 'new_label': 'age', 'dtype': 'float'},
        {'old_label': 'Nudesc_AWR_60min_Delir_ohne_Missings', 'new_label': 'delir_60min', 'dtype': 'int64'},
        {'old_label': 'Nudesc_OP_Tag_Delir', 'new_label': 'delir_day1', 'dtype': 'int64'},
        {'old_label': 'aufPropofol', 'new_label': 'med_auf_propofol', 'dtype': 'int64'},
        {'old_label': 'aufIsoflurane', 'new_label': 'med_auf_isoflurane', 'dtype': 'int64'},
        {'old_label': 'aufDesflurane', 'new_label': 'med_auf_desflurane', 'dtype': 'int64'},
        {'old_label': 'aufSevoflurane', 'new_label': 'med_auf_sevoflurane', 'dtype': 'int64'},
        {'old_label': 'benzodiazepine', 'new_label': 'med_benzodiazepine', 'dtype': 'int64'},
        {'old_label': 'OP_Datum_A', 'new_label': 'op_date', 'dtype': 'datetime64'},
        {'old_label': 'EEG_Durchgefuehrt', 'new_label': 'has_eeg', 'dtype': 'int64'},
        {'old_label': 'Summe_Kristalloide', 'new_label': 'sum_crystalloid', 'dtype': 'float'},
        {'old_label': 'Summe_Kolloide', 'new_label': 'sum_colloid', 'dtype': 'float'},
        {'old_label': 'Glucose_max', 'new_label': 'glucose_max', 'dtype': 'float'},
        {'old_label': 'Glucose_min', 'new_label': 'glucose_min', 'dtype': 'float'},
        {'old_label': 'Hb_max', 'new_label': 'hb_max', 'dtype': 'float'},
        {'old_label': 'Hb_min', 'new_label': 'hb_min', 'dtype': 'float'},
        {'old_label': 'Temperatur_max', 'new_label': 'temp_max', 'dtype': 'float'},
        {'old_label': 'Temperatur_min', 'new_label': 'temp_min', 'dtype': 'float'},
        {'old_label': 'BIS_angewendet', 'new_label': 'has_bis', 'dtype': 'int64'},
        {'old_label': 'S_Beginn', 'new_label': 'op_starttime', 'dtype': 'object'},
        {'old_label': 'S_Ende', 'new_label': 'op_endtime', 'dtype': 'object'},
        {'old_label': 'eindexamethason', 'new_label': 'med_ein_dexamethason', 'dtype': 'int64'},
        {'old_label': 'einsetrone', 'new_label': 'med_ein_setrone', 'dtype': 'int64'},
        {'old_label': 'eindimenhydrinat', 'new_label': 'med_ein_dimenhydrinat', 'dtype': 'int64'},
        {'old_label': 'eindroperisol', 'new_label': 'med_ein_droperisol', 'dtype': 'int64'},
        {'old_label': 'einmcp', 'new_label': 'med_ein_mcp', 'dtype': 'int64'},
        {'old_label': 'exdexamethason', 'new_label': 'med_ex_dexamethason', 'dtype': 'int64'},
        {'old_label': 'exsetrone', 'new_label': 'med_ex_setrone', 'dtype': 'int64'},
        {'old_label': 'exdimenhydrinat', 'new_label': 'med_ex_dimenhydrinat', 'dtype': 'int64'},
        {'old_label': 'exdroperisol', 'new_label': 'med_ex_droperisol', 'dtype': 'int64'},
        {'old_label': 'exmcp', 'new_label': 'med_ex_mcp', 'dtype': 'int64'},
        {'old_label': 'ausdexamethason', 'new_label': 'med_aus_dexamethason', 'dtype': 'int64'},
        {'old_label': 'aussetrone', 'new_label': 'med_aus_setrone', 'dtype': 'int64'},
        {'old_label': 'ausdimenhydrinat', 'new_label': 'med_aus_dimenhydrinat', 'dtype': 'int64'},
        {'old_label': 'ausdroperisol', 'new_label': 'med_aus_droperisol', 'dtype': 'int64'},
        {'old_label': 'ausmcp', 'new_label': 'med_aus_mcp', 'dtype': 'int64'},
        {'old_label': 'einlPropofol', 'new_label': 'med_einl_propofol', 'dtype': 'int64'},
        {'old_label': 'einlThiopental', 'new_label': 'med_einl_thiopental', 'dtype': 'int64'},
        {'old_label': 'einlEtomidate', 'new_label': 'med_einl_etomidate', 'dtype': 'int64'},
        {'old_label': 'einlIsoflurane', 'new_label': 'med_einl_isoflurance', 'dtype': 'int64'},
        {'old_label': 'einlSevoflurane', 'new_label': 'med_einl_sevoflurane', 'dtype': 'int64'},
        {'old_label': 'einlDesflurane', 'new_label': 'med_einl_desflurane', 'dtype': 'int64'},
        {'old_label': 'einlNarkotikaAndere', 'new_label': 'med_einl_narcotics_other', 'dtype': 'int64'},
        {'old_label': 'einlFentanyl', 'new_label': 'med_einl_fentanyl', 'dtype': 'int64'},
        {'old_label': 'einlRemifentanyl', 'new_label': 'med_einl_remifentanyl', 'dtype': 'int64'},
        {'old_label': 'einlAlfentanil', 'new_label': 'med_einl_alfentanil', 'dtype': 'int64'},
        {'old_label': 'einlSufentanyl', 'new_label': 'med_einl_sufentanyl', 'dtype': 'int64'},
        {'old_label': 'einlOpiateAndere', 'new_label': 'med_einl_opioids_other', 'dtype': 'int64'},
        {'old_label': 'aufAndere', 'new_label': 'med_auf_others', 'dtype': 'int64'},
        {'old_label': 'aufRemifentanyl', 'new_label': 'med_auf_remifentanyl', 'dtype': 'int64'},
        {'old_label': 'aufFentanyl', 'new_label': 'med_auf_fentanyl', 'dtype': 'int64'},
        {'old_label': 'aufAlfentanil', 'new_label': 'med_auf_alfentanil', 'dtype': 'int64'},
        {'old_label': 'aufSufentanyl', 'new_label': 'med_auf_sufentanyl', 'dtype': 'int64'},
        {'old_label': 'flumazenil', 'new_label': 'med_flumanzenil', 'dtype': 'int64'},
        {'old_label': 'naloxon', 'new_label': 'med_naloxon', 'dtype': 'int64'},
        {'old_label': 'saggamadex', 'new_label': 'med_saggamadex', 'dtype': 'int64'},
        {'old_label': 'neostigmin', 'new_label': 'med_neostigmin', 'dtype': 'int64'}
    ],
    'Biocog': [

    ]
}

