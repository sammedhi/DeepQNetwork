import numpy as np
import random as rnd
from copy import copy

BLACK           = "\033[0;30m"
DARK_GREY       = "\033[1;30m"
RED             = "\033[0;31m"
LIGHT_RED       = "\033[1;31m"
GREEN           = "\033[0;32m"
LIGHT_GREEN     = "\033[1;32m"
ORANGE          = "\033[0;33m"
YELLOW          = "\033[1;33m"
BLUE            = "\033[0;34m"
LIGHT_BLUE      = "\033[1;34m"
PURPLE          = "\033[0;35m"
LIGHT_PURPLE    = "\033[1;35m"
CYAN            = "\033[0;36m"
LIGHT_CYAN      = "\033[1;36m"
LIGHT_GREY      = "\033[0;37m"
WHITE           = "\033[1;37m"

COLOR_LIST = [WHITE  , YELLOW , LIGHT_BLUE , LIGHT_GREEN]
RESET_COLOR = WHITE
CLEAR_RIGHT = "\033[K"

class DIR:
    UP = 0 ; DOWN = 1 ; RIGHT = 2 ; LEFT = 3

class TYPE:
    EMPTY = 0.0; POINT = 1.0; PLAYER = 9.0

class Game:
    ACTION_NUMBER = 4

    def __init__(self, size=(10 , 10), object_rate = 0.5 , step_size=2 , life=200):
        self.start_life = life
        self.life = life
        self.board = np.full(size , TYPE.EMPTY)
        self.r = 0
        self.step_size = step_size
        self.score = 0
        self.size = size
        self.object_rate = object_rate
        nb_obj = int(size[0] * size[1] * object_rate)
        idx = [[i , j] for i in range(size[0]) for j in range(size[1])]

        rnd.shuffle(idx)
        idx = idx[:nb_obj + 1]
        self.player_pos = idx[0]
        self.board[self.player_pos[0] , self.player_pos[1]] = TYPE.PLAYER

        for i in range(1 , len(idx)):
            self.board[idx[i][0] , idx[i][1]] = TYPE.POINT

        self.s = copy(self.board)
        for i in range(step_size - 1):
            self.s = np.concatenate( (self.s , self.board) , 2)

    def reset(self):
        self.life = self.start_life
        self.board = np.full(self.board.shape , TYPE.EMPTY)
        self.r = 0
        self.score = 0
        nb_obj = int(self.size[0] * self.size[1] * self.object_rate)
        idx = [[i , j] for i in range(self.size[0]) for j in range(self.size[1])]

        rnd.shuffle(idx)
        idx = idx[:nb_obj + 1]
        self.player_pos = idx[0]
        self.board[self.player_pos[0] , self.player_pos[1]] = TYPE.PLAYER

        for i in range(1 , len(idx)):
            self.board[idx[i][0] , idx[i][1]] = TYPE.POINT

        self.s = copy(self.board)
        for i in range(step_size - 1):
            self.s = np.concatenate( (self.s , self.board) , 2)

    def is_terminate(self):
        return self.life == 0

    def step(self , dir):
        self.board[self.player_pos[0] , self.player_pos[1]] = TYPE.EMPTY
        if dir == DIR.UP:
            self.player_pos[1] += 1
        if dir == DIR.DOWN:
            self.player_pos[1] -= 1
        if dir == DIR.LEFT:
            self.player_pos[0] -= 1
        if dir == DIR.RIGHT:
            self.player_pos[0] += 1

        self.player_pos[0] = max(min(self.player_pos[0] , self.board.shape[0] - 1) , 0)
        self.player_pos[1] = max(min(self.player_pos[1] , self.board.shape[1] - 1) , 0)
        r = self.grab_point()
        self.board[self.player_pos[0] , self.player_pos[1]] = TYPE.PLAYER

        return r

    def grab_point(self):
        x , y = self.player_pos
        if self.board[x , y] == TYPE.POINT:
            self.board[x , y] = TYPE.EMPTY
            return 1
        return 0

    def move(self , dir):
        self.r = 0
        self.s = None
        for i in range(self.step_size):
            self.r += self.step(dir)
            if self.s is None:
                self.s = copy(self.board)
            else:
                self.s = np.concatenate((self.s , self.board) , 2)
        self.life -= 1
        self.score += self.r

    def show(self):
        print('\n')
        for j in range(self.board.shape[1]):
            for i in range(self.board.shape[0]):
                if self.board[i , j] == TYPE.PLAYER:
                    print('{0}P{1}'.format(RED , RESET_COLOR) , end='')
                elif self.board[i , j] == 0:
                    print('{0}{2}{1}'.format(GREEN , RESET_COLOR , int(self.board[i , j])) , end='')
                else:
                    print('{0}'.format(int(self.board[i , j])) , end='')
            print()
        print('\n')

    def act(self , action):
        self.move(action)



if __name__ == "__main__":
    import tensorflow as tf
    from deepQ import DeepQ
    from training import Training

    board_size = [16 , 16 , 1]
    step_size = 1

    game = Game(size=board_size , step_size = step_size , object_rate = 0.9)
    #board = np.reshape(game.board , board_size)
    nn = DeepQ(board_size[:2] + [step_size] , 4)


    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    action = sess.run(nn.output ,
                      feed_dict={nn.s:[board]} )
    print(action)
    exit(0)
    '''

    training = Training(nn , game , exp_mem_size = 10000 , learning_rate=0.001 , step_number_greedy_stop=100000 , min_greedy = 0.00)
    for i in range(100):
        for k in range(1000):
            training.next_step()
        print('step : ' , i)

        training.train_batch(32 , 2000)
