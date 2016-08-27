from context import composer
from composer.core import Composer
import numpy as np
import cv2

camera_params_path = 'data/camera_params_matlab.npz'
camera_params = np.load(camera_params_path)
intr_mat = camera_params['intrinsic_matrix']
dstr_co = camera_params['distortion_coeff']
shape = (3000,4000)
c = Composer()
print(c)
c.set_rectification_params(intr_mat, dstr_co, shape)
print(c)
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
left_img, right_img = c.rectify_images(img_left_org, img_right_org)
print(left_img)
cv2.imwrite('test333.jpg',left_img)
c.set_rotation_parameters()
left_img = c.rotate_left(left_img)
cv2.imwrite('test222.jpg',left_img)
right_img = c.rotate_right(right_img)
c.set_homography(left_img, right_img)
