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
        """Set the camera intrinsic and extrinsic parameters."""
        self.intrinsic_matrix = camera_params['intrinsic_matrix']
        self.distortion_coeff = camera_params['distortion_coeff']

    def compose(self, left_img, right_img):
        """Compose both images to panorama."""
        self.estimateTransform(left_img, right_img)
        return self.composePanorama()

    def estimateTransform(self, left_img, right_img):
        """Determine the Transformation matrix for both images."""
        # estimates the image transformation of the left and right image
        self.left_img = left_img
        self.right_img = right_img
        self._rectify_images()
        self._rotate_images()

        adj = Adjuster(self.left_img, self.right_img)
        self.left_trans, self.right_trans, self.hor_l = adj.adjust()

    def _rectify_images(self):
        """Undistort the images by the give camera parameters"""
        self.left_img = imgtools.rectify_img(
            self.left_img, self.intrinsic_matrix, self.distortion_coeff)
        self.right_img = imgtools.rectify_img(
            self.right_img, self.intrinsic_matrix, self.distortion_coeff)

    def _rotate_images(self):
        self.left_img = imgtools.rotate_image(self.left_img, self.left_rot_angle)
        self.right_img = imgtools.rotate_image(self.right_img, self.right_rot_angle)

    def composePanorama(self):
        # get origina width and height of images
        left_h, left_w = self.left_img.shape[:2]
        print("left_h = " + str(left_h))
        print(left_w)
        right_h, right_w = self.right_img.shape[:2]
        left_corners = np.float32([
            [0,         0],
            [0,         left_h],
            [left_w,    left_h],
            [left_w,    0]
            ]).reshape(-1, 1, 2)
        right_corners = np.float32([
            [0,         0],
            [0,         right_h],
            [right_w,   right_h],
            [right_w,   0]
        ]).reshape(-1, 1, 2)

        # transform the corners of the images, to get the dimension of the
        # transformed images and stitched image
        left_corners_trans = cv2.perspectiveTransform(left_corners, self.left_trans)
        right_corners_trans = cv2.perspectiveTransform(right_corners, self.right_trans)
        pts = np.concatenate((left_corners_trans, right_corners_trans), axis=0)

        # measure the max values in x and y direction to get the translation vector
        # so that whole image will be shown
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # define translation matrix
        trans_m = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
        total_size = (xmax - xmin, ymax - ymin)
        result_right = cv2.warpPerspective(self.right_img, trans_m.dot(self.right_trans), total_size)
        self.left_img = cv2.warpPerspective(self.left_img, trans_m.dot(self.left_trans), total_size)

        # unify both layers
        result_right[:total_size[1], :int(self.hor_l + t[0])
            ] = self.left_img[:total_size[1], :int(self.hor_l + t[0])]
    # self.left_img[:total_size[1], int(self.hor_l + t[0]):total_size[0]
    #      ] = self.right[:total_size[1], int(self.hor_l + t[0]):total_size[0]]
        return result_right

if __name__ == "__main__":
    composer = Composer()
    img_left_org = cv2.imread(
        './20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
    img_right_org = cv2.imread(
        './20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
    print(img_left_org.shape)
    camera_params_path = 'camera_params_matlab.npz'
    camera_params = np.load(camera_params_path)
    composer.set_camera_params(camera_params)
    test = composer.compose(img_left_org, img_right_org)
    cv2.imwrite("result.png", test)
