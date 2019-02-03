import cv2
import os
import glob

root_dir = '/home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx'

all_images_name = sorted(glob.glob(root_dir + '/*.png'))

images = {}
index = {}

for img_name in all_images_name:
    img_path = img_name
    img = cv2.imread(img_path)
    images[img_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # compute histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[img_name] = hist

source_images = glob.glob('/home/tliao4/Desktop/kitti_3d_demo_images/*.png')

source_index = {}
for source_img_path in source_images:
    img = cv2.imread(source_img_path)
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    source_index[source_img_path] = hist

# start matching
test_source_img_path, test_source_hist = source_index.items()[9]
results = {}
for (k, hist) in index.items():
    d = cv2.compareHist(test_source_hist, hist, cv2.HISTCMP_CORREL)
    results[k] = d

# sort results
results = sorted([(v, k) for (k, v) in results.items()], reverse = True)

print(results)

# t_img = cv2.imread(test_source_img_path)

nn = results[0][1]

temp = cv2.imread(nn)
print(nn)
cv2.imshow('temp', temp)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 1 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_118.png
# 2 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_1827.png
# 3 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_310.png
# 4 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_197.png
# 5 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_152.png
# 6 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_343.png
# 7 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_1633.png
# 8 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_93.png
# 9 /home/tliao4/Desktop/kitti3d/kitti3d/gt_bbx/car_id_1599.png