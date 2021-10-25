import tensorflow as tf


def optimizer(learner, loss, learning_rate, momentum=0.9):
    opt = None
    if "adagrad" in learner.lower(): 
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate,\
                     initial_accumulator_value=1e-8).minimize(loss)
    elif "rmsprop" in learner.lower():
        opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif "adam" in learner.lower():
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif "gd" in learner.lower():
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  
    elif "momentum" in learner.lower():
        opt = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)  
    else :
        raise ValueError("please select a suitable optimizer")  
    return opt
