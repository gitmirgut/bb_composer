import argparse
import cv2
import numpy as np

from composer.core import Composer


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


if __name__ == '__main__':
    main()
