# TODO rename imgtools to imagetools and import as imgt
import composer.imgtools as imgtools
import composer.point_picker as point_picker
from composer.point_picker import Point_Picker
import cv2
from logging import getLogger
import numpy as np

log = getLogger(__name__)


class Composer(object):

    def __init__(self, rot_angle_l=90, rot_angle_r=-90):
        self.intr_mat = None
        self.dstr_co = None
        self.rot_angle_l = rot_angle_l
        self.rot_angle_r = rot_angle_r
        self.hor_l = None
        self.homo_mat_l = None
        self.homo_mat_r = None

    def create_from_file(file):
        '''Create new Composer with data loaded from file'''
        # TODO error wenn file nicht gefunden
        c = Composer()
        with np.load(file) as data:
            c.rot_angle_l = data['rot_angle_l']
            c.rot_angle_r = data['rot_angle_r']
            c.intr_mat = data['intrinsic_matrix']
            c.dstr_co = data['distortion_coeff']
            c.homo_mat_l = data['homo_mat_l']
            c.homo_mat_r = data['homo_mat_r']
            c.hor_l = data['hor_l']
        log.info('New Composer created from file {}'.format(file))
        log.debug(c)
        return c

    def __repr__(self):
        return('{}(\nintrinsic=\n{},\ndist_coeff\t= {},\nrot_angle_l\t= {},\n'
               'rot_angle_r\t= {},\nhomo_mat_l =\n{},\nhomo_mat_r =\n{}\n)'
               .format(self.__class__.__name__,
                       self.intr_mat,
                       self.dstr_co,
                       self.rot_angle_l,
                       self.rot_angle_r,
                       self.homo_mat_l,
                       self.homo_mat_r))

    def set_camera_params(self, camera_params):
        """Set the camera intrinsic and extrinsic parameters."""
        self.intr_mat = camera_params['intrinsic_matrix']
        self.dstr_co = camera_params['distortion_coeff']
        log.info('Camera parameters loaded.')

    def compose(self, left_img, right_img):
        """Compose both images to panorama."""
        left_img, right_img = self.estimate_transform(left_img, right_img)
        return self.compose_panorama(left_img, right_img)

    def estimate_transform(self, left_img, right_img):
        """Determine the Transformation matrix for both images."""
        # estimates the image transformation of the left and right image
        left_img, right_img = self._rectify_images(left_img, right_img)
        log.info('Images were rectified.')
        left_img, right_img = self._rotate_images(left_img, right_img)
        log.info('Images were rotated.')

        log.info('Point Picker will be initialised.')
        adj = Point_Picker(left_img, right_img)
        quadri_left, quadri_right = adj.pick()
        log.info('Points were picked.')
        log.debug('\nquadri_left =\n{}\nquadri_right =\n{}'.format(quadri_left,quadri_right))
        rect_dest, self.hor_l = point_picker.find_rect(quadri_left, quadri_right)
        log.info('Both homographys have been found.')
        self.homo_mat_l, self.homo_mat_r = point_picker.find_homographys(quadri_left, quadri_right, rect_dest)
        return left_img, right_img

    def _rectify_images(self, left_img, right_img):
        """Undistort the images by the give camera parameters"""
        left_img = imgtools.rectify_img(
            left_img, self.intr_mat, self.dstr_co)
        right_img = imgtools.rectify_img(
            right_img, self.intr_mat, self.dstr_co)
        return left_img, right_img

    def _rotate_images(self, left_img, right_img):
        left_img, self.left_rot_mat = imgtools.rotate_image(
            left_img, self.rot_angle_l)
        right_img,  self.right_rot_mat = imgtools.rotate_image(
            right_img, self.rot_angle_r)
        return left_img, right_img

    def _rotate_points(self, points, left=True):
        if left:
            return cv2.transform(points, self.left_rot_mat)
        else:
            return cv2.transform(points, self.right_rot_mat)

    def compose_panorama(self, left_img, right_img):
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
            left_corners, self.homo_mat_l)
        right_corners_trans = cv2.perspectiveTransform(
            right_corners, self.homo_mat_r)
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
            right_img, trans_m.dot(self.homo_mat_r), total_size)
        left_img = cv2.warpPerspective(
            left_img, trans_m.dot(self.homo_mat_l), total_size)
        self.homo_mat_r = trans_m.dot(self.homo_mat_r)
        self.homo_mat_l = trans_m.dot(self.homo_mat_l)
        # unify both layers
        result_right[:total_size[1], :int(self.hor_l + t[0])
                     ] = left_img[:total_size[1], :int(self.hor_l + t[0])]
        return result_right

    def map_coordinates(self, pts, left=True):
        pts = imgtools.rectify_pts(
            pts, self.intr_mat, self.dstr_co)
        pts = self._rotate_points(pts, left)
        return(pts)
