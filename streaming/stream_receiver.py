import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import imagezmq

from face_mask_detection import FaceMaskDetector
from streaming.streamer import Streamer


class StreamReceiver(threading.Thread):

    def __init__(self, lock, streamers):
        threading.Thread.__init__(self)
        self.daemon = True
        self.lock = lock
        self.streamers = streamers
        self.face_mask_detector = FaceMaskDetector()
        self.executor = ThreadPoolExecutor(8)

    def run(self):
        image_hub = imagezmq.ImageHub()
        print("[INFO] Started listening at " + get_network_device_ip())

        while True:
            (host_name, frame) = image_hub.recv_image()
            frame = cv2.imdecode(frame, 1)
            image_hub.send_reply(b'OK')

            with self.lock:
                streamer = self.get_streamer(host_name)
                if streamer is None:
                    streamer = Streamer(host_name)
                    self.streamers.append(streamer)
                    print("[INFO] Receiving data from {}...".format(host_name))

                self.executor.submit(self.process_frame, streamer, frame)

    def process_frame(self, streamer, frame):
        with self.lock:
            self.face_mask_detector.get_annotated_frame(frame)
            streamer.update_frame(frame)

    def get_streamer(self, host_name):
        for streamer in self.streamers:
            if streamer.host_name == host_name:
                return streamer


def get_network_device_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip
