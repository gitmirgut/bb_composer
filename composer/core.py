# TODO rename imgtools to imagetools and import as imgt
import composer.helpers as helpers
import composer.point_picker as point_picker
from composer.point_picker import Point_Picker
import cv2
from logging import getLogger
import numpy as np

log = getLogger(__name__)


class Composer(object):
    """Compose two images or transform points to a composed area/image."""

    # TODO kann angle raus?
    def __init__(self, rot_angle_l=90, rot_angle_r=-90):
        self.intr_mat = None
        self.dstr_co = None
        self.new_cam_mat = None

        self.rot_mat_l = None
        self.rot_mat_r = None
        self.rot_size_l = None
        self.rot_size_r = None
        self.hor_l = None

        self.homo_mat_l = None
        self.homo_mat_r = None
        self.total_size = None
        self.trans_m = None

    def save_arguments(self, path):
        np.savez(path,
                 intr_mat=self.intr_mat,
                 dstr_co=self.dstr_co,
                 new_cam_mat=self.new_cam_mat,
                 rot_mat_l=self.rot_mat_l,
                 rot_mat_r=self.rot_mat_r,
                 rot_size_l=self.rot_size_l,
                 rot_size_r=self.rot_size_r,
                 hor_l=self.hor_l,
                 homo_mat_l=self.homo_mat_l,
                 homo_mat_r=self.homo_mat_r,
                 total_size=self.total_size,
                 trans_m=self.trans_m
                 )
        log.info('Composer arguments saved to {}'.format(path))
        log.debug('rot_type after save {}'.format(type(self.rot_size_l)))

    def load_arguments(self, path):
        with np.load(path) as data:
            self.intr_mat = data['intr_mat']
            self.dstr_co = data['dstr_co']
            self.new_cam_mat = data['new_cam_mat']
            self.rot_mat_l = data['rot_mat_l']
            self.rot_mat_r = data['rot_mat_r']
            self.rot_size_l = tuple(data['rot_size_l'])
            self.rot_size_r = tuple(data['rot_size_r'])
            self.hor_l = data['hor_l']
            self.homo_mat_l = data['homo_mat_l']
            self.homo_mat_r = data['homo_mat_r']
            self.total_size = tuple(data['total_size'])
            self.trans_m = data['trans_m']
        log.info('Composer arguments loaded from file {}'.format(path))
        log.debug(self)
        log.debug('rot_type after load {}'.format(type(self.rot_size_l)))

    def __repr__(self):
        return('{}(\nintrinsic=\n{},\ndist_coeff\t= {},\nhomo_mat_l =\n{},\nhomo_mat_r =\n{},\nnew_cam_mat=\n{})'
               .format(self.__class__.__name__,
                       self.intr_mat,
                       self.dstr_co,
                       self.homo_mat_l,
                       self.homo_mat_r,
                       self.new_cam_mat))

    def set_rectification_params(self, intr_mat, dstr_co, shape=(3000, 4000)):
        """Set & determine special args for rectification of imgs or points."""
        self.intr_mat = intr_mat
        self.dstr_co = dstr_co
        self.new_cam_mat = helpers.get_cam_mat(intr_mat, dstr_co, shape)

    def rectify_images(self, *images):
        """Rectifiy images.

        Returns a list of rectified images, except if just one image as
        argument is passed, then the return value is just an image.
        """
        # checks if arguments list is empty
        if not images:
            log.warning('List of images for rectification is empty.')
            return None

        # rectify each image of images
        rect_imgs = []
        for img in images:
            rect_imgs.append(cv2.undistort(img, self.intr_mat,
                                           self.dstr_co, None, self.new_cam_mat))

        # if just one argument is passed, return just the one rectified image
        if len(rect_imgs) == 1:
            return rect_imgs[0]

        return rect_imgs

    def rectify_points(self, points):
        """Rectifiy points.

        Returns a list of rectified points, except if just one point as
        argument is passed, then the return value is just an point.
        """
        # TODO check when empty
        return cv2.undistortPoints(
            points, self.intr_mat, self.dstr_co, None, self.new_cam_mat)

    def set_rotation_parameters(self, angle_l=90, angle_r=-90, shape=(3000, 4000)):
        """Determine special arguments for rotation of images and points."""
        self.rot_mat_l, self.rot_size_l = helpers.get_rot_params(
            angle_l, shape)
        self.rot_mat_r, self.rot_size_r = helpers.get_rot_params(
            angle_r, shape)

    def rotate_img(self, *images, rot_mat=None, rot_size=None):
        """Rotate images by given rot_mat and rot_size."""
        # checks if arguments list is empty
        if not images:
            log.warning('No images rotated.')
            return None

        # loads default rotation arguments, when given ones are None
        if rot_mat is None:
            rot_mat = self.rot_mat_l
            rot_size = self.rot_size_l
        log.debug('rot_size = {}'.format(type(rot_size)))

        # rotate each image of images
        rot_imgs = []
        for img in images:
            log.debug('{}\n{}\n{}'.format(img.shape, rot_mat, rot_size))
            rot_imgs.append(cv2.warpAffine(
                img,
                rot_mat,
                rot_size,
                flags=cv2.INTER_LINEAR
            ))

        # if just one argument is passed, return just rectified image
        if len(rot_imgs) == 1:
            return rot_imgs[0]

        return rot_imgs

    def rotate_img_l(self, *images):
        """Rotate image with the arguments for the left image."""
        return self.rotate_img(*images, rot_mat=self.rot_mat_l, rot_size=self.rot_size_l)

    def rotate_img_r(self, *images):
        """Rotate image with the arguments for the right image."""
        return self.rotate_img(*images, rot_mat=self.rot_mat_r, rot_size=self.rot_size_r)

    def rotate_pts(self, pts, rot_mat=None):
        if rot_mat is None:
            rot_mat = self.rot_mat_l
        return cv2.transform(pts, rot_mat)

    def rotate_pts_l(self, pts):
        return self.rotate_pts(pts, rot_mat=self.rot_mat_l)

    def rotate_pts_r(self, pts):
        return self.rotate_pts(pts, rot_mat=self.rot_mat_r)

    def set_couple_parameters(self, img_l, img_r):
        """Determine special arguments for coupling."""
        log.info('Point Picker will be initialised.')
        adj = Point_Picker(img_l, img_r)
        quadri_left, quadri_right = adj.pick()
        log.info('Points were picked.')
        log.debug('\nquadri_left =\n{}\nquadri_right =\n{}'.format(
            quadri_left, quadri_right))
        rect_dest, self.hor_l = helpers.find_rect(
            quadri_left, quadri_right)
        log.info('Both homographys have been found.')
        homo_mat_l, homo_mat_r = helpers.find_homographys(
            quadri_left, quadri_right, rect_dest)
        self.trans_m, self.total_size = helpers.get_translation(
            img_l.shape[:2], img_r.shape[:2], homo_mat_l, homo_mat_r)
        self.homo_mat_r = self.trans_m.dot(homo_mat_r)
        self.homo_mat_l = self.trans_m.dot(homo_mat_l)

    def couple_pano(self, img_l, img_r):
        """Unite both images."""
        # warp left image to dst format
        result_r = cv2.warpPerspective(
            img_r, self.homo_mat_r, self.total_size)
        # warp right image to dst format
        result_l = cv2.warpPerspective(img_l, self.homo_mat_l, self.total_size)

        # unite
        result_r[:, :int(self.hor_l + self.trans_m[0][2])
                 ] = result_l[:, :int(self.hor_l + self.trans_m[0][2])]
        return result_r

    def apply_homography(self, pts, homo=None):
        if homo is None:
            homo = self.homo_mat_l
        return cv2.perspectiveTransform(pts, homo)

    def apply_homography_l(self, pts):
        return self.apply_homography(pts, self.homo_mat_l)

    def apply_homography_r(self, pts):
        return self.apply_homography(pts, self.homo_mat_r)
