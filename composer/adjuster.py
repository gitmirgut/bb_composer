from draggable_marker import DraggableMarker
import matplotlib.pyplot as plt
import numpy as np

class Adjuster(object):
    def __init__(self, left_img, right_img):
        self.left_img = left_img
        self.right_img = right_img

    def adjust(self):

        def _on_click(event):
            if event.button == 1 and event.dblclick:
                if event.inaxes == ax_left:
                    add_draggable_marker(event, ax_left, dms_left)
                elif event.inaxes == ax_right:
                    add_draggable_marker(event, ax_right, dms_right)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, tight_layout=True)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        # TODO remove alpha channel when exist for display
        ax_left.imshow(self.left_img)
        ax_right.imshow(self.right_img)
        dms_left = []
        dms_right = []
        c_id = fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
        assert(len(dms_left)==len(dms_right))


def add_draggable_marker(event, axis, dms):
    print('x = ' + str(event.xdata) + '| y = ' + str(event.xdata))
    marker, = axis.plot(event.xdata, event.ydata, 'xr', markersize=20)

    # initialize draggable marker that is initialized with a Marker but
    # will move its x,y location when dragged
    dm = DraggableMarker(marker)
    dm.connect()
    dms.append(dm)
    plt.show()

def to_orderd_list(dms_list):
    ms = np.zeros((1, len(dms_list), 2), np.float32)
    for i in range(len(dms_list)):
        ms[0][i] = dms_list[i].mark.get_xydata()[0]
    # TODO hier geht es weiter vlg. transform2
    # corners_marked_l = np.zeros((1, 4, 2), np.float32)
    # corners_marked_l[0][0]=ms_left[0][order(ms_left[0])[0]]
    # corners_marked_l[0][1]=ms_left[0][order(ms_left[0])[1]]
    # corners_marked_l[0][2]=ms_left[0][order(ms_left[0])[2]]
    # corners_marked_l[0][3]=ms_left[0][order(ms_left[0])[3]]
