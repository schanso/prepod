import mne

import prepod.lib.io as io
import prepod.lib.constants as const

layout = io.construct_layout()
picks = [idx for idx, el in enumerate(layout.names) if el in const.CH_NAMES]
mne.viz.plot_layout(layout, picks)
