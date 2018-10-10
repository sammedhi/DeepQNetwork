import tensorflow as tf
import utilNN

class DeepQ:
    def __init__(self , input_shape , action_number):
        self.input = tf.placeholder(tf.float32 , shape=[None] + list(input_shape))
        self.labels = tf.placeholder(tf.float32 , shape=[None , action_number])

        ################################# CONV LAYERS ##########################################
        with tf.variable_scope("CONV1"):
            conv1 = utilNN.conv(self.input ,
                                kernel_size = 8 ,
                                nb_kernel = 32,
                                in_channel = input_shape[2],
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
        ################################# CONV LAYERS ##########################################
        shape_input_fc = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
        fc_input = tf.manip.reshape(conv3 , [-1 , shape_input_fc])

        with tf.variable_scope("FC1"):
            fc1 , W , B = utilNN.fc_layer(fc_input , [shape_input_fc , 512] , tf.nn.tanh)
        with tf.variable_scope("FC2"):
            self.out , W , B = utilNN.fc_layer(fc1 , [512 , action_number])

        self.output = tf.nn.softmax(self.out)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.input,
                                                                           labels = self.labels))
