from context import composer
from composer.composer import Composer
import numpy as np
import cv2

c = Composer()
camera_params_path = 'data/camera_params_matlab.npz'
camera_params = np.load(camera_params_path)
c.set_camera_params(camera_params)
test = np.zeros((10, 1, 2), dtype=np.float32)
# test_o ut = imgtools.rectify_pts(test, c.intrinsic_matrix, c.distortion_coeff)
# print(test_out[test_out < 0]) TODO problematisch wenn < 0 ?
# print(c.map_coordinates(test))
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
test = c.compose(img_left_org, img_right_org)

cv2.imwrite("result.png", test)
np.savez('composer_params.npz',
         left_rot_angle=c.left_rot_angle,
         right_rot_angle=c.right_rot_angle,
         intrinsic_matrix=c.intrinsic_matrix,
         distortion_coeff=c.distortion_coeff,
         hor_l = c.hor_l,
         left_trans=c.left_trans,
         right_trans=c.right_trans)
npzfile = np.load('composer_params.npz')
print(npzfile.files)
