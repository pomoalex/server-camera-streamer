# USAGE
# python server.py

import math
import time
from datetime import datetime
from threading import Thread, Lock

import cv2
import imagezmq
import imutils
from imutils import build_montages

from utils import get_network_device_ip

frame_dict = {}
live_clients = {}
LIVENESS_CHECK_SECONDS = 5
MAX_INACTIVITY = 3


def display_montage():
    global frame_dict
    frames = list(frame_dict.values())
    count = len(frames)
    if count > 0:
        # TODO: check h, w
        (h, w) = frames[0].shape[:2]
        rows = math.ceil(math.sqrt(count))
        columns = math.floor(math.sqrt(count))
        montage = build_montages(frames, (w, h), (rows, columns))[0]
        cv2.imshow("Remote camera streaming", montage)


def add_frame_to_frame_dict(frame, host_name):
    frame = imutils.resize(frame, width=500)

    cv2.putText(frame, host_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame_dict[host_name] = frame


def check_liveness(lock):
    while True:
        time.sleep(LIVENESS_CHECK_SECONDS)
        with lock:
            for (host_name, start_time) in list(live_clients.items()):
                if (datetime.now() - start_time).seconds > MAX_INACTIVITY:
                    print("[INFO] lost connection to {}".format(host_name))
                    live_clients.pop(host_name)
                    frame_dict.pop(host_name)


def receive_streams(lock):
    image_hub = imagezmq.ImageHub()

    print("[INFO] Started listening at " + get_network_device_ip())

    while True:
        (host_name, frame) = image_hub.recv_image()
        image_hub.send_reply(b'OK')

        with lock:
            if host_name not in live_clients.keys():
                print("[INFO] receiving data from {}...".format(host_name))

            live_clients[host_name] = datetime.now()
            add_frame_to_frame_dict(frame, host_name)


def serve_streams():
    lock = Lock()

    stream_receiver_thread = Thread(target=receive_streams, args=(lock,))
    # daemon threads are terminated after main thread dies
    stream_receiver_thread.daemon = True
    stream_receiver_thread.start()

    liveness_check_thread = Thread(target=check_liveness, args=(lock,))
    # daemon threads are terminated after main thread dies
    liveness_check_thread.daemon = True
    liveness_check_thread.start()

    while True:
        with lock:
            display_montage()

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    serve_streams()
