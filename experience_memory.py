import random as rnd
import numpy as np

class ExperienceMemory:
    def __init__(self , max_size):
        self.memory = []
        self.index = 0
        self.max_size = max_size

    def push(self , s , s_ , r , a):
        episode = [s , s_ , r , a ]
        if len(self.memory) < self.max_size:
            self.memory.append(episode)
        else:
            self.memory[self.index] = episode
            self.index = (self.index + 1) % self.max_size

    def pick_random(self , nb_sample):
        idx = np.arange(len(self.memory))
        rnd.shuffle(idx)
        idx = idx[:nb_sample]
        return {'s' : [self.memory[i][0] for i in idx] ,
                's_' : [self.memory[i][1] for i in idx] ,
                'r' : [self.memory[i][2] for i in idx] ,
                'a' : [self.memory[i][3] for i in idx] }
