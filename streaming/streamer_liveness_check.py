import threading
import time
from datetime import datetime


class StreamerLivenessCheck(threading.Thread):

    def __init__(self, lock, streamers):
        threading.Thread.__init__(self)
        self.daemon = True
        self.LIVENESS_CHECK_SECONDS = 5
        self.MAX_INACTIVITY = 3
        self.lock = lock
        self.streamers = streamers

    def run(self):
        while True:
            time.sleep(self.LIVENESS_CHECK_SECONDS)
            with self.lock:
                for streamer in self.streamers:
                    if (datetime.now() - streamer.last_alive).seconds > self.MAX_INACTIVITY:
                        print("[INFO] Lost connection to {}".format(streamer.host_name))
                        self.streamers.remove(streamer)
