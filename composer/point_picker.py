import matplotlib.pyplot as plt

from composer import helpers
from composer.draggable_marker import add_draggable_marker
from composer.draggable_marker import dms_to_pts


class PointPicker(object):
    """GUI for picking points."""

    def __init__(self, img_l, right_img):
        self.img_l = img_l
        self.right_img = right_img
        self.count_dms_left = 0
        self.count_dms_right = 0

    def pick(self):
        """Initialise GUI to pick 4 points on each side.

        A matplot GUI will be initialised, where the user has to pick 4 points
        on the left and right image. Afterwards the PointPicker will return 2
        clockwise sorted list of the picked points.
        """

        def _on_click(event):
            if event.button == 1 and event.dblclick:
                if event.inaxes == ax_left and len(dms_left) < 4:
                    self.count_dms_left += 1
                    add_draggable_marker(
                        event, ax_left, dms_left, self.img_l)
                elif event.inaxes == ax_right and len(dms_right) < 4:
                    self.count_dms_right += 1
                    add_draggable_marker(
                        event, ax_right, dms_right, self.right_img)

        fig, (ax_left, ax_right) = plt.subplots(
            nrows=1, ncols=2, tight_layout=True)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        # TODO remove alpha channel when exist for display
        ax_left.imshow(self.img_l)
        ax_right.imshow(self.right_img)
        dms_left = set()
        dms_right = set()
        # TODO c_id
        c_id = fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
        assert ((len(dms_left) == 4) and (len(dms_right) == 4))
        quadri_left = helpers.sort_pts(dms_to_pts(dms_left))
        quadri_right = helpers.sort_pts(dms_to_pts(dms_right))

        return quadri_left, quadri_right
