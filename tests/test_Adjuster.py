from context import composer
from composer.core import Composer
import cv2
import logging.config
import numpy as np

# handler = logging.StreamHandler(sys.stdout)
logging.config.fileConfig('logging_config.ini')


c = Composer()
camera_params_path = 'data/camera_params_matlab.npz'
camera_params = np.load(camera_params_path)
c.set_camera_params(camera_params)
test = np.zeros((10, 1, 2), dtype=np.float32)
# test_o ut = imgtools.rectify_pts(test, c.intr_mat, c.dstr_coeff)
# print(test_out[test_out < 0]) TODO problematisch wenn < 0 ?
# print(c.map_coordinates(test))
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
test = c.compose(img_left_org, img_right_org)

cv2.imwrite("result.png", test)
np.savez('composer_params.npz',
         rot_angle_l=c.rot_angle_l,
         rot_angle_r=c.rot_angle_r,
         intr_mat=c.intr_mat,
         dstr_coeff=c.dstr_co,
         hor_l = c.hor_l,
         homo_mat_l=c.homo_mat_l,
         homo_mat_r=c.homo_mat_r)
npzfile = np.load('composer_params.npz')
print(npzfile.files)
