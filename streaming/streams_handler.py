import math
from threading import Lock

import cv2
import numpy as np
from imutils import build_montages

from streaming.stream_receiver import StreamReceiver
from streaming.streamer_liveness_check import StreamerLivenessCheck


class StreamsHandler:
    def __init__(self):
        self.stream_lock = Lock()
        self.frame_dict = {}
        self.live_streamers = {}
        self.stream_receiver_thread = StreamReceiver(self.stream_lock, self.frame_dict, self.live_streamers)
        self.liveness_check_thread = StreamerLivenessCheck(self.stream_lock, self.frame_dict, self.live_streamers)

    def start_handling(self):
        self.stream_receiver_thread.start()
        self.liveness_check_thread.start()

    def build_montage(self):
        with self.stream_lock:
            frames = list(self.frame_dict.values())
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
