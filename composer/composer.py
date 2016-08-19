import cv2
# TODO rename imgtools to imagetools and import as imgt
import imgtools
from adjuster import Adjuster
import numpy as np

class Composer(object):

    def __init__(self, left_rot_angle=90, right_rot_angle=-90):
        self.left_img = None
        self.right_img = None
        self.intrinsic_matrix = None
        self.distortion_coeff = None
        self.left_rot_angle = left_rot_angle
        self.right_rot_angle = right_rot_angle
        self.left_trans = None
        self.right_trans = None

    def set_camera_params(self, camera_params):
        self.intrinsic_matrix = camera_params['intrinsic_matrix']
        self.distortion_coeff = camera_params['distortion_coeff']

    def compose(self, left_img, right_img, pano):
        status = self.estimateTransform(left_img, right_img)
        if (status is not True):
            return status
        return self.composePanorama(pano)

    def estimateTransform(self, left_img, right_img):
        # estimates the image transformation of the left and right image
        self.left_img = left_img
        self.right_img = right_img
        self._rectify_images()
        self._rotate_images()
        adj = Adjuster(self.left_img, self.right_img)
        adj.adjust()

    def _rectify_images(self):
        self.left_img = imgtools.rectify_img(
            self.left_img, self.intrinsic_matrix, self.distortion_coeff)
        self.right_img = imgtools.rectify_img(
            self.right_img, self.intrinsic_matrix, self.distortion_coeff)

    def _rotate_images(self):
        self.left_img = imgtools.rotate_image(self.left_img, self.left_rot_angle)
        self.right_img = imgtools.rotate_image(self.right_img, self.right_rot_angle)

    def composePanorama(self, pano):
        # compose the left and right image
        pass

if __name__ == "__main__":
    composer = Composer()
    img_left_org = cv2.imread(
        './20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
    img_right_org = cv2.imread(
        './20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
    camera_params_path = 'camera_params_matlab.npz'
    camera_params = np.load(camera_params_path)
    composer.set_camera_params(camera_params)
    composer.estimateTransform(img_left_org,img_right_org)
    cv2.imshow('image',composer.left_img)
    cv2.waitKey(500)

    # test = Composer()
    # test()
    # print(test.left_img)
    # print(test2.left_img)
    # print(test2.camera_params)
