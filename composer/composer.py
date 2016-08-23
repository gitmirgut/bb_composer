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
        self.hor_l = None
        self.left_trans = None
        self.right_trans = None

    def create_from_file(file):
        c = Composer()
        with np.load(file) as data:
            c.left_rot_angle = data['left_rot_angle']
            c.right_rot_angle = data['right_rot_angle']
            c.intrinsic_matrix = data['intrinsic_matrix']
            c.distortion_coeff = data['distortion_coeff']
            c.left_trans = data['left_trans']
            c.right_trans = data['right_trans']
            c.hor_l = data['hor_l']
        return c

    def __repr__(self):
        return('{}(\nintrinsic=\n{},\ndist_coeff\t= {},\nleft_rot_angle\t= {},\n'
               'right_rot_angle\t= {},\nleft_trans =\n{},\nright_trans =\n{}\n)'
               .format(self.__class__.__name__,
                       self.intrinsic_matrix,
                       self.distortion_coeff,
                       self.left_rot_angle,
                       self.right_rot_angle,
                       self.left_trans,
                       self.right_trans))

    def set_camera_params(self, camera_params):
        """Set the camera intrinsic and extrinsic parameters."""
        self.intrinsic_matrix = camera_params['intrinsic_matrix']
        self.distortion_coeff = camera_params['distortion_coeff']


    def compose(self, left_img, right_img):
        """Compose both images to panorama."""
        left_img, right_img = self.estimateTransform(left_img, right_img)
        print(self)
        # self.left_img, self.right_img = left_img, right_img
        return self.composePanorama(left_img, right_img)

    def estimateTransform(self, left_img, right_img):
        """Determine the Transformation matrix for both images."""
        # estimates the image transformation of the left and right image
        # self.left_img = left_img
        # self.right_img = right_img
        left_img, right_img = self._rectify_images(left_img, right_img)
        left_img, right_img = self._rotate_images(left_img, right_img)

        adj = Adjuster(left_img, right_img)
        self.left_trans, self.right_trans, self.hor_l = adj.adjust()
        return left_img, right_img

    def _rectify_images(self, left_img, right_img):
        """Undistort the images by the give camera parameters"""
        left_img = imgtools.rectify_img(
            left_img, self.intrinsic_matrix, self.distortion_coeff)
        right_img = imgtools.rectify_img(
            right_img, self.intrinsic_matrix, self.distortion_coeff)
        return left_img, right_img

    def _rotate_images(self, left_img, right_img):
        left_img = imgtools.rotate_image(left_img, self.left_rot_angle)
        right_img = imgtools.rotate_image(right_img, self.right_rot_angle)
        return left_img, right_img

    def composePanorama(self, left_img, right_img):
        # get origina width and height of images
        left_h, left_w = left_img.shape[:2]
        right_h, right_w = right_img.shape[:2]
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
        left_corners_trans = cv2.perspectiveTransform(
            left_corners, self.left_trans)
        right_corners_trans = cv2.perspectiveTransform(
            right_corners, self.right_trans)
        pts = np.concatenate((left_corners_trans, right_corners_trans), axis=0)

        # measure the max values in x and y direction to get the translation vector
        # so that whole image will be shown
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # define translation matrix
        trans_m = np.array(
            [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
        total_size = (xmax - xmin, ymax - ymin)
        result_right = cv2.warpPerspective(
            right_img, trans_m.dot(self.right_trans), total_size)
        left_img = cv2.warpPerspective(
            left_img, trans_m.dot(self.left_trans), total_size)

        # unify both layers
        result_right[:total_size[1], :int(self.hor_l + t[0])
                     ] = left_img[:total_size[1], :int(self.hor_l + t[0])]
        return result_right

    def map_coordinates(self, pts, cam_side=None):
        pts = imgtools.rectify_pts(
            pts, self.intrinsic_matrix, self.distortion_coeff)
        return(pts)

if __name__ == "__main__":

    # c = Composer()
    # camera_params_path = 'camera_params_matlab.npz'
    # camera_params = np.load(camera_params_path)
    # c.set_camera_params(camera_params)
    # test = np.zeros((10, 1, 2), dtype=np.float32)
    # # test_out = imgtools.rectify_pts(test, c.intrinsic_matrix, c.distortion_coeff)
    # # print(test_out[test_out < 0]) TODO problematisch wenn < 0 ?
    # print(c.map_coordinates(test))
    # img_left_org = cv2.imread(
    #     './20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
    # img_right_org = cv2.imread(
    #     './20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
    # test = c.compose(img_left_org, img_right_org)
    #
    # cv2.imwrite("result.png", test)
    # np.savez('composer_params.npz',
    #          left_rot_angle=c.left_rot_angle,
    #          right_rot_angle=c.right_rot_angle,
    #          intrinsic_matrix=c.intrinsic_matrix,
    #          distortion_coeff=c.distortion_coeff,
    #          hor_l = c.hor_l,
    #          left_trans=c.left_trans,
    #          right_trans=c.right_trans)
    # npzfile = np.load('composer_params.npz')
    # print(npzfile.files)
    img_left_org = cv2.imread(
        './20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
    img_right_org = cv2.imread(
        './20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
    nc = Composer.create_from_file('composer_params.npz')
    print('rect')
    left_img, right_img = nc._rectify_images(img_left_org, img_right_org)
    print('rot')
    left_img, right_img = nc._rotate_images(left_img, right_img)
    print('compose')
    result = nc.composePanorama(left_img, right_img)
    cv2.imwrite("loaded_result.png", result)
    print(nc)
