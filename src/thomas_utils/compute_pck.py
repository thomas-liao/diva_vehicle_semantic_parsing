import os
import glob
import numpy as np
import cv2

alpha = 0.1 # for alpha @ PCK
# gt_n = 'gt_occ{}_images'
gt_n = 'det_occ{}_images'



# pred_n = 'hg_os23d_gt_occ{}_160k'
# pred_n = 'L2_os23d/L2_os23d_gt_occ{}'
# pred_n = 'hg_23d/hg_23d_gt_occ{}'
# pred_n = 'L2_os23d_pmc/L2_os23d_gt_padding_occ{}'
# pred_n = 'L2_os23d_pmc_real/L2_os23d_gt_padding_occ{}'  L2_os23d_pmc_real/L2_os23d_gt_padding_occ{}_80k
pred_n = 'L2_os23d_pmc_real/L2_os23d_gt_padding_occ{}_200k'

base_dir = '/home/tliao4/Desktop/kitti3d/kitti3d/tliao4'
gt_dirs = [os.path.join(base_dir, f) for f in [gt_n.format(i) for i in range(0, 4)]]
pred_dirs = [os.path.join(base_dir, f) for f in [pred_n.format(i) for i in range(0, 4)]]

all_accur = []
for gt_dir, pred_dir in zip(gt_dirs, pred_dirs):
    gt_names = sorted(glob.glob(gt_dir + '/*.2d'))
    gt_names = [n for n in gt_names]
    pred_names = [n[:-3].replace(gt_dir, pred_dir) + '_2d.txt' for n in gt_names]
    accur_pool = []

    for pred_name, gt_name in zip(pred_names, gt_names):
        h, w, _ = cv2.imread(gt_name.replace('.2d', '.png')).shape
        h *= 1.0
        w *= 1.0

        k2d_pred = np.loadtxt(pred_name)
        k2d_pred = k2d_pred.reshape((36, 2))
        k2d_pred[:, 0] *= w
        k2d_pred[:, 1] *= h
        k2d_gt = np.loadtxt(gt_name)
        k2d_gt = np.transpose(k2d_gt.reshape((2, 36)))

        # diag = np.linalg.norm((h, w))
        # thres = alpha * diag
        thres = alpha * 1.0 * max(h, w)

        # now compare
        # acc = []
        # for idx in range(k2d_gt.shape[0]):
        #     if k2d_gt[idx][1] >= h or k2d_gt[idx][1]>= w or k2d_gt[idx][0] < 0 or k2d_gt[idx][0] < 0:
        #         continue
        #     err = np.linalg.norm(k2d_pred[idx] - k2d_gt[idx])
        #     if err >= thres:
        #         acc.append(0.0)
        #     else:
        #         acc.append(1.0)
        # if len(acc) == 0:
        #     continue
        # temp = np.mean(acc)
        # if temp == 0:
        #     aabbcc = 1
        # accur_pool.append(np.mean(acc))

        # weighted averaging
        for idx in range(k2d_gt.shape[0]):
            if k2d_gt[idx][1] >= h or k2d_gt[idx][1] >= w or k2d_gt[idx][0] < 0 or k2d_gt[idx][0] < 0:
                continue
            err = np.linalg.norm(k2d_pred[idx] - k2d_gt[idx])
            if err > thres:
                accur_pool.append(0.0)
            else:
                accur_pool.append(1.0)

    all_accur.append(np.mean(accur_pool))

print(all_accur)

a = 0



