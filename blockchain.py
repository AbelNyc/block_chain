#import the necessary libraries
from hashlib import sha256
from datetime import datetime


class Block():
    def __init__(self,transactions,previous_hash,nonce=0):
        #initialize the attributes
        self.transactions=transactions
        self.previous_hash=previous_hash
        self.nonce=nonce
        self.timestamp=datetime.now()
        self.hash = self.generate_hash()
    def print_block(self):
         # print the contents in the block
         print("Timestamp is :", self.timestamp)
         print("Transactions:", self.transactions)
         print("Current Hash: ", self.generate_hash())
    def generate_hash(self):
        #hash the block contents
        block_contents=str(self.timestamp)+str(self.transactions)+str(self.previous_hash)+str(self.nonce)
        hash_block=sha256(block_contents.encode())
        return hash_block.hexdigest()
