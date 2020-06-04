import socket
import threading
from datetime import datetime

import cv2
import imagezmq
import imutils


class StreamReceiver(threading.Thread):

    def __init__(self, lock, frame_dict, live_clients):
        threading.Thread.__init__(self)
        self.daemon = True
        self.lock = lock
        self.frame_dict = frame_dict
        self.live_clients = live_clients

    def run(self):
        image_hub = imagezmq.ImageHub()

        print("[INFO] Started listening at " + get_network_device_ip())

        while True:
            (host_name, frame) = image_hub.recv_image()
            image_hub.send_reply(b'OK')

            with self.lock:
                if host_name not in self.live_clients.keys():
                    print("[INFO] receiving data from {}...".format(host_name))

                self.live_clients[host_name] = datetime.now()
                self.add_frame_to_frame_dict(frame, host_name)

    def add_frame_to_frame_dict(self, frame, host_name):
        frame = imutils.resize(frame, width=500)

        cv2.putText(frame, host_name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
