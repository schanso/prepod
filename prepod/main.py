import numpy as np
from wyrm.types import Data

from prepod.lib.io import store_raws, import_folder, import_targets, append_labels
import prepod.lib.globals as glob
from prepod.lib.models import train_test_wyrm, lda_vyrm

proj_folder = '/Users/jannes/Projects/delir/data/sudocu/'
dir_in = proj_folder + 'brainvision/raw/'
dir_out = dir_in + 'npy/wyrm/frontal/'
target_path = proj_folder + 'subject_data.csv'

# dir_in = '/Users/jannes/Projects/delir/data/biocog/brainvision/intra/raw/'
# dir_out = dir_in + 'npy/wyrm/frontal/'

# store_raws(dir_in=dir_in, dir_out=dir_out, ftype_in='edf', out_format='wyrm', subset='frontal')
data = import_folder(dir_in=dir_out, in_format='wyrm', exclude=['2170_2170_R4.npy'])
targets = import_targets(fpath=target_path,
                         colname_subjid=glob.COLNAME_SUBJID_SUDOCU,
                         colname_target=glob.COLNAME_TARGET_SUDOCU)
dat = append_labels(data, targets)
data = []
classes = []
names = dat[0].names
units = dat[0].units
for el in dat:
    data.append(el.data)
    classes.append(el.axes[0])
axes = [np.array(classes), dat[0].axes[1], dat[0].axes[2]]

dat = Data(data=data, axes=axes, names=names, units=units)
dat_train, dat_test = train_test_wyrm(dat, test_size=.3)
acc = lda_vyrm(data_train=dat_train, data_test=dat_test)
print(acc)

#
#     path_targets = '/Users/jannes/Projects/delir/data/sudocu/subject_data.csv'
#     labels = import_targets(fpath=path_targets,
#                             subj_ids=subj_ids)
#     datasets = append_labels(datasets, labels)
#
#     data = []
#     classes = []
#     names = datasets[0].names
#     units = datasets[0].units
#     for el in datasets:
#         data.append(el.data)
#         classes.append(el.axes[0])
#     data = np.array(data).squeeze()
#     axes = [np.array(classes), datasets[0].axes[1], datasets[0].axes[2]]
#
#     dat = Data(data=data, axes=axes, names=names, units=units)
#     np.save(path_out, dat)
#
# dat = np.load(path_out)
# dat = dat.flatten()[0]
# dat2 = feature_vector(dat)
# dat_train, dat_test = train_test_wyrm(dat2, test_size=.3)
# clf = lda_train(dat_train)
# out = lda_apply(dat_test, clf)
# res = (np.sign(out) + 1) / 2
# print((res == dat_test.axes[0]).sum() / len(res))
