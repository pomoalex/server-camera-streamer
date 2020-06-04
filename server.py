# USAGE
# python server.py

import cv2
from flask import Flask
from flask import Response
from flask import render_template

from streaming import StreamsHandler

app = Flask(__name__)
streamsHandler = StreamsHandler()


def serve_streams():
    while True:
        output_frame = streamsHandler.build_montage()
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

        if not flag:
            continue

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
    streamsHandler.start_handling()
    app.run(host='0.0.0.0', port=80, debug=True,
            threaded=True, use_reloader=False)
