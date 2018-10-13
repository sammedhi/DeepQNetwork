import tensorflow as tf
import utilNN
import numpy as np

class DeepQ:
    def __init__(self , input_shape , action_number , gamma = 0.8):
        self.s = tf.placeholder(tf.float32 , shape=[None] + list(input_shape) , name='s')
        self.s_ =  tf.placeholder_with_default( np.zeros([1] + input_shape , dtype="float32") , shape=[None] + input_shape , name='s_')
        self.r =  tf.placeholder_with_default( np.zeros((1) , dtype="float32") , shape=[None] , name='r')
        self.a = tf.placeholder_with_default( np.zeros((1 , action_number), dtype="float32") , shape=[None , action_number] , name='a')
        self.input_shape = input_shape
        self.action_number = action_number

        self.out = self.get_output(self.s)
        self.out_ = self.get_output(self.s_ , reuse=True)
        self.output = tf.one_hot (tf.argmax(self.out , 1) , action_number)

        self.max_q_ = tf.reduce_max(self.out_ , 1) * gamma + self.r
        self.max_q_ = tf.stack([self.max_q_ for i in range(action_number)] , axis=1)
        self.cost = tf.reduce_mean( tf.squared_difference( tf.multiply(self.out , self.a ) , tf.multiply(tf.stop_gradient(self.max_q_) , self.a) ) )

        tf.summary.histogram("R" , self.r)
        tf.summary.histogram("action" , tf.argmax(self.out , 1))



    def get_output(self , inp , reuse=False):
        ################################# CONV LAYERS ##########################################
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with tf.variable_scope("CONV1"):
                conv1 = utilNN.conv(self.s ,
                                    kernel_size = 8 ,
                                    nb_kernel = 32,
                                    in_channel = self.input_shape[2],
                                    strides = [1 , 4 , 4 , 1])

            with tf.variable_scope("CONV2"):
                conv2 = utilNN.conv(conv1 ,
                                    kernel_size = 4 ,
                                    nb_kernel = 64,
                                    in_channel = 32,
                                    strides = [1 , 2 , 2 , 1])

            with tf.variable_scope("CONV3"):
                conv3 = utilNN.conv(conv2 ,
                                    kernel_size = 3 ,
                                    nb_kernel = 64,
                                    in_channel = 64,
                                    strides = [1 , 1 , 1 , 1])
            ################################# FC LAYERS ##########################################
            shape_input_fc = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
            fc_input = tf.manip.reshape(conv3 , [-1 , shape_input_fc])

            with tf.variable_scope("FC1"):
                fc1 , W , B = utilNN.fc_layer(fc_input , [shape_input_fc , 512] , tf.nn.relu)
            with tf.variable_scope("FC2"):
                out , W , B = utilNN.fc_layer(fc1 , [512 , self.action_number])

            return out
