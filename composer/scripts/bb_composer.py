import argparse
import cv2
import numpy as np

from composer.core import Composer

# composer = Composer()
# img_left_org = cv2.imread(
#     '../20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
# img_right_org = cv2.imread(
#     '../20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
# print(img_left_org.shape)
# camera_params_path = 'camera_params_matlab.npz'
# camera_params = np.load(camera_params_path)
# composer.set_camera_params(camera_params)
# test = composer.compose(img_left_org, img_right_org)
# cv2.imwrite("result.png", test)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook composer',
        description='Determine the Parameter for the Coupling of two images.')

    
