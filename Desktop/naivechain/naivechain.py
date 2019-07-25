import nacl.hash
import websockets
import json
import os

http = os.environ.get()

class block:
    def __init__(self, index, previous_hash, time_stamp, data, hash):
        self.index = index
        self.previous_hash = str(previous_hash)
        self.time_stamp = time_stamp
        self.data = data
        self.hash = str(hash)

sockets = []


