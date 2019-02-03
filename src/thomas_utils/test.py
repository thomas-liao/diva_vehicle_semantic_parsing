import cv2
import numpy as np

img = cv2.imread('/home/tliao4/Desktop/kitti3d/kitti3d/tliao4/gt_occ0_images/car_id_2.png')
h, w, _ = img.shape
pred_2d = np.loadtxt('/home/tliao4/Desktop/kitti3d/kitti3d/tliao4/hg_os23d_gt_occ0/car_id_2_2d.txt')

pred_2d = pred_2d.reshape((36, 2))

pred_2d[:, 0] *= w
pred_2d[:, 1] *= h

for (x, y) in pred_2d[18:, :]:
    cv2.circle(img, (int(x), int(y)), 2, [0, 0, 255], -1)

cv2.imshow('temp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
