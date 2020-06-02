# USAGE
# python server.py

import math
from datetime import datetime

import cv2
import imagezmq
import imutils
from imutils import build_montages

from utils import get_network_device_ip

image_hub = imagezmq.ImageHub()
frame_dict = {}
live_clients = {}
last_liveness_check = datetime.now()
LIVENESS_CHECK_SECONDS = 5

print("[INFO] Started listening at " + get_network_device_ip() + ":5555")

while True:
    (host_name, frame) = image_hub.recv_image()
    image_hub.send_reply(b'OK')

    if host_name not in live_clients.keys():
        print("[INFO] receiving data from {}...".format(host_name))

    live_clients[host_name] = datetime.now()

    frame = imutils.resize(frame, width=500)

    cv2.putText(frame, host_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame_dict[host_name] = frame

    (h, w) = frame.shape[:2]

    rows = math.ceil(math.sqrt(len(frame_dict)))
    columns = math.floor(math.sqrt(len(frame_dict)))
    montages = build_montages(frame_dict.values(), (w, h), (rows, columns))

    cv2.imshow("Remote camera streaming", montages[0])

    if (datetime.now() - last_liveness_check).seconds > LIVENESS_CHECK_SECONDS:
        for (host_name, start_time) in list(live_clients.items()):
            if (datetime.now() - start_time).seconds > LIVENESS_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(host_name))
                live_clients.pop(host_name)
                frame_dict.pop(host_name)

        last_liveness_check = datetime.now()

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
