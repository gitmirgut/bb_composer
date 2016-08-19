class DraggableMarker(object):
    lock = None  # only one can be animated at a time

    def __init__(self, mark):
        self.mark = mark
        # self.mark.set_color('y')
        self.press = None
        self.background = None

    def connect(self):
        '''connect to all the events we need'''
        self.c_id_press = self.mark.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.c_id_release = self.mark.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.c_id_motion = self.mark.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store
        some data
        """
        if event.inaxes != self.mark.axes:
            return
        if DraggableMarker.lock is not None:
            return
        contains, attrd = self.mark.contains(event)
        if not contains:
            return
        print('event contains', self.mark.get_xydata()[0])
        x, y = self.mark.get_xydata()[0]
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
        'on motion we will move the mark if the mouse is over us'
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
        'on release we reset the press data'
        if DraggableMarker.lock is not self:
            return

        self.press = None
        DraggableMarker.lock = None

        # turn off the mark animation property and reset the background
        self.mark.set_animated(False)
        self.background = None

        # redraw the full figure
        self.mark.figure.canvas.draw()

    def on_enter(self):
        self.mark.set_color('y')
        self.mark.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.mark.figure.canvas.mpl_disconnect(self.c_id_press)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_release)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_motion)
        self.mark.figure.canvas.draw()
