import logging.config
import os.path

import cv2
import numpy as np
import composer.core
# from composer.core import Composer

logging.config.fileConfig('logging_config.ini')

camera_params_path = 'data/test_structure/Input/camera_params_matlab.npz'
img_l = 'data/test_structure/Input/Cam_0_20160715130847_631282.jpg'
img_r = 'data/test_structure/Input/Cam_1_20160715130847_631282.jpg'
camera_params = np.load(camera_params_path)
intr_mat = camera_params['intrinsic_matrix']
dstr_co = camera_params['distortion_coeff']
shape = (3000, 4000)
c = composer.core.Composer()
c.set_rectification_params(intr_mat, dstr_co, shape)
c.set_meta_data(img_l, img_r)
img_left_org = cv2.imread(img_l)
print(img_left_org)
img_right_org = cv2.imread(img_r)
left_img, right_img = c.rectify_images(img_left_org, img_right_org)
c.set_rotation_parameters()
left_img = c.rotate_img_l(left_img)
right_img = c.rotate_img_r(right_img)
c.set_couple_parameters(left_img, right_img)
result = c.couple_pano(left_img, right_img)
if not os.path.exists('./data/test_structure/Output'):
    os.mkdir('./data/test_structure/Output')
cv2.imwrite('./data/test_structure/Output/result.jpg', result)
c.save_arguments('./data/test_structure/Output/composer_data.npz')
print(c.camIdx_l)
