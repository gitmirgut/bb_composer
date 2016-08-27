from context import composer
from composer.core import Composer
import cv2
import numpy as np
import logging.config

logging.config.fileConfig('logging_config.ini')

camera_params_path = 'data/camera_params_matlab.npz'
camera_params = np.load(camera_params_path)
intr_mat = camera_params['intrinsic_matrix']
dstr_co = camera_params['distortion_coeff']
shape = (3000, 4000)
c = Composer()
c.set_rectification_params(intr_mat, dstr_co, shape)
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
left_img, right_img = c.rectify_images(img_left_org, img_right_org)
c.set_rotation_parameters()
left_img = c.rotate_img_l(left_img)
right_img = c.rotate_img_r(right_img)
c.set_couple_parameters(left_img, right_img)
result = c.couple_pano(left_img, right_img)
cv2.imwrite('new.jpg', result)
