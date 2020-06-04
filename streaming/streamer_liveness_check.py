import threading
import time
from datetime import datetime


class StreamerLivenessCheck(threading.Thread):

    def __init__(self, lock, frame_dict, live_streamers):
        threading.Thread.__init__(self)
        self.daemon = True
        self.LIVENESS_CHECK_SECONDS = 5
        self.MAX_INACTIVITY = 3
        self.lock = lock
        self.frame_dict = frame_dict
        self.live_streamers = live_streamers

    def run(self):
        while True:
            time.sleep(self.LIVENESS_CHECK_SECONDS)
            with self.lock:
                for (host_name, start_time) in list(self.live_streamers.items()):
                    if (datetime.now() - start_time).seconds > self.MAX_INACTIVITY:
                        print("[INFO] lost connection to {}".format(host_name))
                        self.live_streamers.pop(host_name)
                        self.frame_dict.pop(host_name)
