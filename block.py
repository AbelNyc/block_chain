from blockchain import Block
class Blockchain():
    def __init__(self):
        self.chain=[]
        self.all_transactions=[]
        self.genesis_block()
    def genesis_block(self):
        transactions={}
        previous_hash="0"
        self.chain.append(Block(transactions,previous_hash))
    def print_blocks(self):
        for x in range(len(self.chain)):
            current_block=self.chain[x]
            print("Block {} {}".format(x,current_block))
            #current_block.print_contents()
            current_block.print_block()

    def New_block(self,transactions):
        #add new block to blochain
        previous_blockhash=self.chain[len(self.chain)-1].hash
        new_block=Block(transactions,previous_blockhash)
        proof = self.proofOfWork(new_block)
        self.chain.append(new_block)
        return proof, new_block

    def verify_chain(self):
        for x in range(1,len(self.chain)):
            current=self.chain[x]
            previous=self.chain[x-1]
            if current.hash!=current.generate_hash():
                print("The Current hash of the block is not equal to the generated hash of the block.")
                return False
            if current.previous_hash!=previous.generate_hash():
                print("The previous block hash is not the equal to the value stored in the current block.")
                return False
        return True

    def proofOfWork(self,block,difficulty=3):
        proof=block.generate_hash()
        while proof[:difficulty]!='0'*difficulty:
            block.nonce+=1
            proof=block.generate_hash()
            block.nonce=0
            return proof