from datetime import datetime


class Streamer:
    def __init__(self, host_name):
        self.host_name = host_name
        self.last_alive = datetime.now()
        self.frame = None

    def update_frame(self, frame):
        self.frame = frame
        self.last_alive = datetime.now()
