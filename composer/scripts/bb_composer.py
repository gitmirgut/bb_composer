import argparse
import cv2
import numpy as np

from composer.core import Composer


def process_images(args):
    img_left_org = cv2.imread(args.left_img_path)
    img_right_org = cv2.imread(args.right_img_path)
    cam_par = np.load(args.cam_params_path)
    c = Composer()
    c.set_meta_data(args.left_img_path, args.right_img_path)
    c.set_rectification_params(cam_par['intrinsic_matrix'],
                               cam_par['distortion_coeff'],
                               img_left_org.shape[:2])
    left_rect, right_rect = c.rectify_images(img_left_org, img_right_org)
    c.set_rotation_parameters()
    left_rot = c.rotate_img_l(left_rect)
    right_rot = c.rotate_img_r(right_rect)
    c.set_couple_parameters(left_rot, right_rot)
    result = c.couple_pano(left_rot, right_rot)
    c.save_arguments(args.composing_params_out)
    cv2.imwrite('result.jpg', result)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook composer',
        description='Determine the Parameter for the Coupling of two images.')

    parser.add_argument(
        'left_img_path', help='path of the left image', type=str)
    parser.add_argument(
        'right_img_path', help='path of the right image', type=str)
    parser.add_argument(
        'cam_params_path', help='path of the file which holds'
        ' the camera parameters', type=str)
    parser.add_argument(
        'composing_params_out', help='path of the output file, which will hold'
        ' the coupling parameters'
    )
    args = parser.parse_args()
    process_images(args)


if __name__ == '__main__':
    main()
