import tensorflow as tf
from experience_memory import ExperienceMemory
import numpy as np
import random as rnd

class Training:
    def __init__(self , nn , train_env , exp_mem_size = 200000 , learning_rate = 0.0001 , step_number_greedy_stop=10000 , min_greedy = 0.05):
        self.nn = nn
        self.train_env = train_env
        self.action = 0

        self.sess = tf.Session()
        self.trainer = tf.train.AdamOptimizer(learning_rate).minimize(nn.cost)
        self.sess.run(tf.global_variables_initializer())

        self.mem = ExperienceMemory(exp_mem_size)
        self.greedy_eps = 1
        self.greedy_eps_step = (self.greedy_eps - min_greedy) / step_number_greedy_stop
        self.min_greedy = min_greedy

        self.writer = tf.summary.FileWriter("../summary" , self.sess.graph)
        self.merged_summary = tf.summary.merge_all()

    def next_step(self):
        self.prev_s = self.train_env.s
        rnd_action = np.zeros( (self.train_env.ACTION_NUMBER) )
        rnd_action[rnd.randint(0 , self.train_env.ACTION_NUMBER - 1)] = 1.0
        self.action = rnd_action if rnd.random() < self.greedy_eps else self.sess.run(self.nn.output ,
                                                                                         feed_dict = {self.nn.s: [self.train_env.s] })[0]
        self.train_env.act(np.argmax(self.action))
        self.add_mem()

        if self.greedy_eps > self.min_greedy:
            self.greedy_eps -= self.greedy_eps_step

        if self.train_env.is_terminate():
            print(self.train_env.score)
            self.train_env.reset()


    def train_batch(self , batch_size , frame_train):
        print(self.greedy_eps)
        nb_batch = frame_train // batch_size
        for batch_id in range(nb_batch):
            batch = self.mem.pick_random(batch_size)
            _ , cost , summaries = self.sess.run([self.trainer , self.nn.cost , self.merged_summary] ,
                          feed_dict={self.nn.s : batch['s'], self.nn.s_ : batch['s_'], self.nn.r : batch['r'], self.nn.a : batch['a']})
            self.writer.add_summary(summaries)
            #print('cost : ' , cost)


    def play(self , n_step):
        for i in range(n_step):
            self.next_step()

    def add_mem(self):
        self.mem.push(self.prev_s , self.train_env.s , self.train_env.r , self.action)
