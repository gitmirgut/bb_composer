from logging import getLogger

import cv2
import matplotlib.pyplot as plt
import numpy as np

log = getLogger(__name__)


class DraggableMarker(object):
    """Defines Marker which can be dragged by mouse.

    The placed marker can be dragged by simple left click and can be refined
    by pressing the button.
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    lock = None  # only one can be animated at a time

    def __init__(self, mark, img):
        """Initialize a DraggableMarker, with the img for later refinement."""
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.mark = mark
        self.press = None
        self.background = None

    def connect(self):
        """Connect to all needed Events."""
        self.c_id_press = self.mark.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.c_id_release = self.mark.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.c_id_motion = self.mark.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.c_id_key = self.mark.figure.canvas.mpl_connect(
            'key_release_event', self.on_key)

    def on_press(self, event):
        """Check on button press if mouse is over this DraggableMarker."""
        if event.inaxes != self.mark.axes:
            return
        if DraggableMarker.lock is not None:
            return

        # This checks if the mouse is over us (marker)
        contains, attrd = self.mark.contains(event)
        if not contains:
            return
        x, y = self.mark.get_xydata()[0]
        self.mark.set_color('r')
        self.press = x, y, event.xdata, event.ydata
        DraggableMarker.lock = self

        # draw everything but the selected marker and store the pixel buffer
        canvas = self.mark.figure.canvas
        axes = self.mark.axes
        self.mark.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.mark.axes.bbox)

        # now redraw just the marker
        axes.draw_artist(self.mark)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """On motion the mark will move if the mouse is over this marker."""
        if DraggableMarker.lock is not self:
            return
        if event.inaxes != self.mark.axes:
            return
        x, y, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.mark.set_xdata(x + dx)
        self.mark.set_ydata(y + dy)

        canvas = self.mark.figure.canvas
        axes = self.mark.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.mark)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """On release the press data will be reset."""
        if DraggableMarker.lock is not self:
            return

        self.press = None
        DraggableMarker.lock = None

        # turn off the mark animation property and reset the background
        self.mark.set_animated(False)
        self.background = None

        # redraw the full figure
        self.mark.figure.canvas.draw()

    def on_key(self, event):
        """Check what key ist pressed and executes corresponding function."""
        if event.inaxes != self.mark.axes:
            return
        contains, attrd = self.mark.contains(event)
        if not contains:
            return
        if event.key == 'b':
            log.info(
                'You pressed {}, the marker will be refined!'.format(event.key))
            xy = np.array([self.mark.get_xydata()[:]])
            log.debug('old coordinates = ' + str(xy))
            xy_new = refine(self.img, xy)
            log.debug('new coordinates = ' + str(xy_new))
            self.mark.set_xdata(xy_new[0][0][0])
            self.mark.set_ydata(xy_new[0][0][1])
            self.mark.set_color('g')
            plt.show()

    def disconnect(self):
        """disconnect all the stored connection ids."""
        self.mark.figure.canvas.mpl_disconnect(self.c_id_press)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_release)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_motion)
        self.mark.figure.canvas.draw()


def dms_to_pts(dms_list):
    """Extract the coordinates of the draggable Markers from a list."""
    dms_list = list(dms_list)
    pts = np.zeros((len(dms_list), 2), np.float32)
    for i, dm in enumerate(dms_list):
        pts[i] = dm.mark.get_xydata()[0]
    return pts


def add_draggable_marker(event, axis, dms, img):
    """Add a DraggableMarker to the axis and to the list dms."""
    log.info('Create draggable Marker.')
    log.debug('x = ' + str(event.xdata) + ' | y = ' + str(event.xdata))
    marker, = axis.plot(event.xdata, event.ydata, 'xr', markersize=20, markeredgewidth=2)

    # initialize draggable marker that is initialized with a Marker but
    # will move its x,y location when dragged
    dm = DraggableMarker(marker, img)
    dm.connect()
    dms.add(dm)
    plt.show()


def refine(img, corner):
    """Refine the corner location."""
    corner_new = np.ones((1, 1, 2), np.float32)
    corner_new[0][0] = corner[0][0][0], corner[0][0][1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(img, corner_new, (40, 40), (-1, -1), criteria)
    return corner_new
