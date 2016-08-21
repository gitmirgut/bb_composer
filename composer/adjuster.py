from draggable_marker import DraggableMarker, dms_to_pts, add_draggable_marker
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Adjuster(object):
    def __init__(self, left_img, right_img):
        self.left_img = left_img
        self.right_img = right_img
        self.count_dms_left = 0
        self.count_dms_right = 0


    def adjust(self):

        def _on_click(event):
            if event.button == 1 and event.dblclick:
                if event.inaxes == ax_left and len(dms_left) < 4:
                    print(self.count_dms_left)
                    self.count_dms_left += 1
                    add_draggable_marker(event, ax_left, dms_left, self.left_img)
                elif event.inaxes == ax_right and len(dms_right) < 4:
                    self.count_dms_right += 1
                    add_draggable_marker(event, ax_right, dms_right, self.right_img)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, tight_layout=True)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        # TODO remove alpha channel when exist for display
        ax_left.imshow(self.left_img)
        ax_right.imshow(self.right_img)
        dms_left = set()
        dms_right = set()
        # TODO c_id
        c_id = fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
        assert(len(dms_left) == len(dms_right))

        quadri_left = sort_pts(dms_to_pts(dms_left))
        quadri_right = sort_pts(dms_to_pts(dms_right))

        """
        The following steps will determine the dimension, of the rectangle/s to
        which the quadrilaterals will be mapped to.
        """

        """
        Rename corners for better orientation.
        ul_l----um_l / um_r----ul_r
         |         |    |        |
         |  left   |    | right  |
         |         |    |        |
        dl_l----dm_l / dm_r----dl_r
        """
        ul_l = quadri_left[0]
        um_l = quadri_left[1]
        dl_l = quadri_left[2]
        dm_l = quadri_left[3]

        ul_r = quadri_right[0]
        um_r = quadri_right[1]
        dl_r = quadri_right[2]
        dm_r = quadri_right[3]

        # get the euclidean distances between the corners of the quadrilaterals
        u_l = np.linalg.norm(ul_l - um_l)
        d_l = np.linalg.norm(dl_l - dm_l)
        l_l = np.linalg.norm(ul_l - dl_l)
        r_l = np.linalg.norm(um_l - dm_l)
        u_r = np.linalg.norm(ul_r - um_r)
        d_r = np.linalg.norm(dl_r - dm_r)
        l_r = np.linalg.norm(ul_r - dl_r)
        r_r = np.linalg.norm(um_r - dm_r)

        hor_l = max(u_l, d_l)
        hor_r = max(u_r, d_r)
        vert = max(l_l, r_l, l_r, r_r)
        print(hor_l)
        print(hor_r)
        print(vert)

        """
        Declare the dimension of the destination rectangle.

        rect_dest:
        0 ----  1 ----- 2
        |       |       |
        | left  | right |
        |       |       |
        5 ----  4 ----- 3
        """

        rect_dest = np.zeros((6, 2), np.float32)
        rect_dest[0] = 0, 0
        rect_dest[1] = hor_l, 0
        rect_dest[2] = hor_l + hor_r, 0
        rect_dest[3] = hor_l + hor_r, vert
        rect_dest[4] = hor_l, vert
        rect_dest[5] = 0, vert

        rect_left = np.array([rect_dest[0],rect_dest[1],rect_dest[4],rect_dest[5]])
        rect_right = np.array([rect_dest[1],rect_dest[2],rect_dest[3],rect_dest[4]])
        left_h, m = cv2.findHomography(quadri_left, rect_left)
        right_h, m = cv2.findHomography(quadri_right, rect_right)
        return left_h, right_h, hor_l

def sort_pts(pts):
    sort_pts = np.zeros((len(pts), 2), np.float32)
    for i in range(len(pts)):
        sort_pts[i] = pts[argsort_pts(pts)[i]]
    return sort_pts

def argsort_pts(points):
    """order points as convex quadrilateral.

    Return the indices that would order the points,
    so that they form a convex quadrilateral.

    points:                quadri:
         x   x                      A---B
                      --->         /     \
       x       x                  D-------C

    """
    assert(len(points) == 4)

    # calculate the barycentre / centre of gravity
    barycentre = np.zeros((1, 2), np.float32)
    barycentre = points.sum(axis=0) / 4

    # var for saving the points in realtion to the barycentre
    bary_vector = np.zeros((4, 2), np.float32)

    # var for saving the closest point of the origin
    A = None
    min_dist = None

    for i in range(len(points)):

        # determine the distance to the origin
        cur_dist_origin = np.linalg.norm(points[i])

        # save the closest point of the orgin
        if A is None or cur_dist_origin < min_dist:
            min_dist = cur_dist_origin
            A = i

        # determine point in relation to the barycentre
        bary_vector[i] = points[i] - barycentre

    angles = np.zeros(4, np.float32)
    # determine the angles of the different points in relation to the line
    # between cloest point of origin (A) and barycentre
    for i in range(len(points)):
        if i != A:
            cur_angle = np.arctan2((np.linalg.det((bary_vector[A], bary_vector[i]))), np.dot(
                bary_vector[A], bary_vector[i]))
            if cur_angle < 0:
                cur_angle = 2 * np.pi + cur_angle
            angles[i] = cur_angle
    return np.argsort(angles)
