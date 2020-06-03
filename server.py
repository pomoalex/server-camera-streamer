# USAGE
# python server.py

import math
from datetime import datetime

import cv2
import imagezmq
import imutils
from imutils import build_montages

from utils import get_network_device_ip

frame_dict = {}
live_clients = {}
last_liveness_check = datetime.now()
LIVENESS_CHECK_SECONDS = 5


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
    global frame_dict
    frame = imutils.resize(frame, width=500)

    cv2.putText(frame, host_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame_dict[host_name] = frame


def check_liveness():
    global last_liveness_check
    if (datetime.now() - last_liveness_check).seconds > LIVENESS_CHECK_SECONDS:
        for (host_name, start_time) in list(live_clients.items()):
            if (datetime.now() - start_time).seconds > LIVENESS_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(host_name))
                live_clients.pop(host_name)
                frame_dict.pop(host_name)

        last_liveness_check = datetime.now()


def everything():
    image_hub = imagezmq.ImageHub()

    print("[INFO] Started listening at " + get_network_device_ip() + ":5555")

    while True:
        (host_name, frame) = image_hub.recv_image()
        image_hub.send_reply(b'OK')

        if host_name not in live_clients.keys():
            print("[INFO] receiving data from {}...".format(host_name))

        live_clients[host_name] = datetime.now()

        add_frame_to_frame_dict(frame, host_name)

        display_montage()

        check_liveness()

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    everything()
