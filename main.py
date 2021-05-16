from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
import numpy as np
import cv2


Builder.load_file("myapplayout.kv")

class AndroidCamera(Camera):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera_resolution = (640, 480)
    face_resolution = (128, 96)
    ratio = camera_resolution[0] / face_resolution[0]


    def _camera_loaded(self, *largs):
        self.texture = Texture.create(size=np.flip(self.camera_resolution), colorfmt='rgb')
        self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None
        frame = self.frame_from_buf()

        self.frame_to_screen(frame)
        super(AndroidCamera, self).on_tex(*l)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.face_det(frame_rgb)

        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tostring()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.face_resolution[1], self.face_resolution[0]))
        faces = self.face_cascade.detectMultiScale(resized, 1.3, 2)
        if len(faces) != 0:
            face = faces[np.argmax(faces[:, 3])]
            x, y, w, h = face
            cv2.rectangle(frame, (int(x * self.ratio), int(y * self.ratio)), (int((x + w) * self.ratio), int((y + h) * self.ratio)), (0, 255, 0), 2)



class MyLayout(BoxLayout):
    pass


class MyApp(App):
    def build(self):
        return MyLayout()


if __name__ == '__main__':
    MyApp().run()
