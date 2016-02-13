from __future__ import division
import tensorflow as tf
import numpy as np
from constants import batch_size, image_size, num_channels, num_classes
from simple_feeder import Feeder
from time import gmtime, strftime

layer1_patch_size = 7
layer1_stride = 2
layer2_patch_size = 5
layer2_stride = 1
layer3_patch_size = 3
layer3_stride = 1

layer1_depth = 24
layer2_depth = 32
layer3_depth = 32
num_hidden = 800
# MOVING_AVERAGE_DECAY = 0.999


def get_var(v, averaged):
    '''
    Returns either the current version of the variable or the moving average version
    depending on averaged
    :param v: tf variable
    :param averaged: true to return the moving average, false to return the current version
    :return: tf variable
    '''
    # if averaged:
    #     return variable_averages.average(v)
    # else:
    #     return v

    # No averaging at the moment
    return v

if __name__ == '__main__':

    feeder = Feeder()

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
        # TODO: Don't build a whole graph for these.
        # Instead run the validation and test data through the same batch process as the train data.
        tf_valid_dataset = tf.constant(feeder.get_data_set('valid')[0])
        tf_test_dataset = tf.constant(feeder.get_data_set('test')[0])

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [layer1_patch_size, layer1_patch_size, num_channels, layer1_depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([layer1_depth]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [layer2_patch_size, layer2_patch_size, layer1_depth, layer2_depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_depth]))

        layer3_weights = tf.Variable(tf.truncated_normal(
            [layer3_patch_size, layer3_patch_size, layer2_depth, layer3_depth], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_depth]))

        layer4_weights = tf.Variable(tf.truncated_normal(
            [1568, num_hidden], stddev=0.1))  # TODO math
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer5_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_classes], stddev=0.1))
        layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))

        layer1_weight_decay = 0.0005
        layer2_weight_decay = 0.0005
        layer3_weight_decay = 0.0005
        layer4_weight_decay = 0.001
        layer5_weight_decay = 0.001

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        # variable_averages_op = variable_averages.apply(tf.trainable_variables())


        # Model.
        def model(data, dropout=False, averaged=False):
            if dropout:
                keep_proba = 0.65
            else:
                keep_proba = 1

            w1 = get_var(layer1_weights, averaged)
            b1 = get_var(layer1_biases, averaged)
            w2 = get_var(layer2_weights, averaged)
            b2 = get_var(layer2_biases, averaged)
            w3 = get_var(layer3_weights, averaged)
            b3 = get_var(layer3_biases, averaged)
            w4 = get_var(layer4_weights, averaged)
            b4 = get_var(layer4_biases, averaged)
            w5 = get_var(layer5_weights, averaged)
            b5 = get_var(layer5_biases, averaged)

            conv = tf.nn.conv2d(data, w1, [1, layer1_stride, layer1_stride, 1], padding='SAME')
            hidden = tf.nn.relu(conv + b1)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
            conv = tf.nn.conv2d(pool, w2, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + b2)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
            conv = tf.nn.conv2d(pool, w3, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + b3)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, w4) + b4)
            dropout = tf.nn.dropout(hidden, keep_proba)
            return tf.matmul(dropout, w5) + b5


        # Training computation.
        logits = model(tf_train_dataset, dropout=True)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        l2_loss = (layer1_weight_decay * tf.nn.l2_loss(layer1_weights)
                   + layer2_weight_decay * tf.nn.l2_loss(layer2_weights)
                   + layer3_weight_decay * tf.nn.l2_loss(layer3_weights)
                   + layer4_weight_decay * tf.nn.l2_loss(layer4_weights)
                   + layer5_weight_decay * tf.nn.l2_loss(layer5_weights))

        loss = cross_entropy + l2_loss

        # Optimizer.
        optimizer = tf.train.MomentumOptimizer(0.0001, 0.01).minimize(loss)

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits)


        def run_epoch(session, data='valid', train=False):
            '''
            Runs through an epoch of data, trains or not depending on train. Prints
            the accuracy and the average cross entropy
            :param data: The data set to use.  One of 'train', 'valid' or 'test'
            :param train: Whether or not to train the model based on the data
            :return: None
            '''

            # number of data points
            n = 0
            # cross entropy combines additively
            total_cross_entropy = 0
            # correctly classified data points
            correct = 0

            for batch_data, batch_labels in feeder.epoch(data):
                n += batch_data.shape[0]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                if train:
                    # Include optimization op because we're training
                    _, l, predictions = session.run(
                        [optimizer,
                         cross_entropy,
                         prediction,
                         ], feed_dict=feed_dict)
                else:
                    # Don't run the optimization op because we're testing not training
                    l, predictions = session.run([cross_entropy, prediction, ], feed_dict=feed_dict)

                total_cross_entropy += l
                correct += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))

            print "%s Completed epoch of %s data set" % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), data)
            print "Accuracy: %.1f" % (correct/n * 100)
            print "Average cross entropy per observation %.3f" % (total_cross_entropy/n)
            print '\n'

    num_steps = 200

    with tf.Session(graph=graph) as session:
        # Friday Goals:
        ## Temporarily ditch ewma
        ## train_epoch and test_epoch functions - same function
        ## run a few train_epoch's and then a test_epoch using the same graph but different feed dicts
        ## report on both the accuracy and cross entropy of both types of epochs

        tf.initialize_all_variables().run()
        print "Initialized"
        for step in xrange(num_steps):
            run_epoch(session, data='train', train=True)
            # The validation epoch is much faster because the optimization op is the most expensive
            # op and because the validation dataset is smaller.
            run_epoch(session, data='valid', train=False)
        run_epoch(session, data='test', train=False)
