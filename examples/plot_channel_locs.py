from datetime import datetime as dt
import mne
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('font', family = 'serif', serif = 'cmr10', size=18)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import prepod.lib.io as io
import prepod.lib.constants as const

path_out = '/Users/jannes/Projects/delir/results/plots/channel_locs/{}_channel_locs.png'.format(dt.now().strftime('%Y%m%d%H%M%S'))

prefrontal_coordinates = [(-0.11, 0.405), (0.11, 0.405)]
frontal_coordinates = [(-0.282, 0.263), (-0.145, 0.23), (0, 0.223), (0.145, 0.23), (0.282, 0.263)]
parietal_coordinates = [(-0.145, -0.185), (0, -0.175), (0.145, -0.185)]
layout = io.construct_layout()
picks = [idx for idx, el in enumerate(layout.names) if el in const.CH_NAMES]
fig, ax = mne.viz.plot_layout(layout, picks, show=False)
for coordinate_tuple in prefrontal_coordinates:
    ax.add_artist(plt.Circle(coordinate_tuple, 0.04, color='orange', alpha=1))
for coordinate_tuple in frontal_coordinates:
    ax.add_artist(plt.Circle(coordinate_tuple, 0.04, color='#999999', alpha=1))
for coordinate_tuple in parietal_coordinates:
    ax.add_artist(plt.Circle(coordinate_tuple, 0.04, color='#fe2151', alpha=1))

prefrontal_patch = mpatches.Patch(color='orange', label='Prefrontal')
frontal_patch = mpatches.Patch(color='#999999', label='Frontal')
parietal_patch = mpatches.Patch(color='#fe2151', label='Parietal')
ax.legend(handles=[prefrontal_patch, frontal_patch, parietal_patch], fontsize=14, loc=4)

plt.savefig(path_out, dpi=300)
