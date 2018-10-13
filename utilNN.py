
import tensorflow as tf
def fc_layer(input , size , act_f=None):
    W = tf.get_variable("W" ,size ,tf.float32, tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("B" ,size[1] ,tf.float32, tf.constant_initializer(0.0))
    if act_f is None:
        act = tf.add(tf.matmul(input , W) , B)
    else:
        act = act_f(tf.add(tf.matmul(input , W) , B) )
    tf.summary.histogram("W" , W)
    tf.summary.histogram("act" , act)
    return act , W , B


def conv(input ,  kernel_size , nb_kernel , in_channel , act=tf.nn.relu , strides = [0 , 1 , 1 , 0]):
    W = tf.get_variable("Wconv" , [kernel_size , kernel_size , in_channel , nb_kernel] , tf.float32 , tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("Bconv" , [nb_kernel] , tf.float32 , tf.constant_initializer(0.0))
    tf.summary.histogram("W" , W)
    return act(tf.add ( tf.nn.conv2d( input , W , strides , "SAME") , B) )

def max_pool2x2(input):
    return  tf.nn.max_pool(input ,
                           [1 , 2 , 2 , 1] ,
                           [0 , 2 , 2 , 0] ,
                           "SAME")
