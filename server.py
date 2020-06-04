# USAGE
# python server.py

from threading import Lock

import cv2
from flask import Flask
from flask import Response
from flask import render_template

from streaming import StreamerLivenessCheck, StreamReceiver

app = Flask(__name__)
stream_lock = Lock()
frame_dict = {}
live_streamers = {}
stream_receiver_thread = StreamReceiver(stream_lock, frame_dict, live_streamers)
liveness_check_thread = StreamerLivenessCheck(stream_lock, frame_dict, live_streamers)


def serve_streams():
    while True:
        with stream_lock:
            output_frame = stream_receiver_thread.build_montage()

            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(serve_streams(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    stream_receiver_thread.start()
    liveness_check_thread.start()
    app.run(host='0.0.0.0', port=80, debug=True,
            threaded=True, use_reloader=False)
