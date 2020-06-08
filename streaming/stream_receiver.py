import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import imagezmq

from face_mask_detection import FaceMaskDetector


class StreamReceiver(threading.Thread):

    def __init__(self, lock, frame_dict, live_clients):
        threading.Thread.__init__(self)
        self.daemon = True
        self.lock = lock
        self.frame_dict = frame_dict
        self.live_clients = live_clients
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
                if host_name not in self.live_clients.keys():
                    print("[INFO] Receiving data from {}...".format(host_name))

                self.live_clients[host_name] = datetime.now()
                self.executor.submit(self.process_frame, frame, host_name)

    def process_frame(self, frame, host_name):
        with self.lock:
            self.face_mask_detector.get_annotated_frame(frame)
            self.frame_dict[host_name] = frame


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
