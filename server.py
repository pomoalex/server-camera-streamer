# USAGE
# python server.py

import math
import time
from datetime import datetime
from threading import Thread, Lock

import cv2
import imagezmq
import imutils
import numpy as np
from flask import Flask
from flask import Response
from flask import render_template
from imutils import build_montages

from utils import get_network_device_ip

frame_dict = {}
live_clients = {}
LIVENESS_CHECK_SECONDS = 5
MAX_INACTIVITY = 3


def add_frame_to_frame_dict(frame, host_name):
    frame = imutils.resize(frame, width=500)

    cv2.putText(frame, host_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame_dict[host_name] = frame


def check_liveness():
    while True:
        time.sleep(LIVENESS_CHECK_SECONDS)
        with stream_lock:
            for (host_name, start_time) in list(live_clients.items()):
                if (datetime.now() - start_time).seconds > MAX_INACTIVITY:
                    print("[INFO] lost connection to {}".format(host_name))
                    live_clients.pop(host_name)
                    frame_dict.pop(host_name)


def receive_streams():
    image_hub = imagezmq.ImageHub()

    print("[INFO] Started listening at " + get_network_device_ip())

    while True:
        (host_name, frame) = image_hub.recv_image()
        image_hub.send_reply(b'OK')

        with stream_lock:
            if host_name not in live_clients.keys():
                print("[INFO] receiving data from {}...".format(host_name))

            live_clients[host_name] = datetime.now()
            add_frame_to_frame_dict(frame, host_name)


def launch_streaming_threads():
    stream_receiver_thread = Thread(target=receive_streams)
    # daemon threads are terminated after main thread dies
    stream_receiver_thread.daemon = True
    stream_receiver_thread.start()

    liveness_check_thread = Thread(target=check_liveness)
    # daemon threads are terminated after main thread dies
    liveness_check_thread.daemon = True
    liveness_check_thread.start()


def build_montage():
    frames = list(frame_dict.values())
    count = len(frames)
    if count > 0:
        (h, w) = frames[0].shape[:2]
        columns = math.ceil(math.sqrt(count))
        rows = columns if count > columns * (columns - 1) else columns - 1
        montage = build_montages(frames, (w, h), (columns, rows))[0]
        return montage.copy()
    else:
        frame = np.zeros(shape=[375, 500, 3], dtype=np.uint8)
        text = 'No camera available'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, 'No camera available', ((500 - text_size[0]) // 2, 375 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame


def serve_streams():
    while True:
        with stream_lock:
            output_frame = build_montage()

            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


app = Flask(__name__)
stream_lock = Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(serve_streams(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    launch_streaming_threads()
    app.run(host='localhost', port=8000, debug=True,
            threaded=True, use_reloader=False)
