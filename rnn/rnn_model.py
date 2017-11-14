from __future__ import print_function
import tensorflow as tf
import numpy as np
import utils
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import os
from datetime import datetime



def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(model, lr, lr_decay, lr_decay_after, grad_clipping, num_hidden, num_layers, batch_size, train_seq_length,
    feature_size = 757,
    num_classes = 3):

    reset_graph()

    # Placeholders
    X = tf.placeholder(tf.float32, [batch_size, train_seq_length, feature_size], name='X')
    Y = tf.placeholder(tf.float32, [batch_size, num_classes], name='Y')

    
    
    seqlen = train_seq_length
    keep_prob = tf.constant(1.0)

    # RNN
    if model == 'lstm':
        cell_fn = rnn.BasicLSTMCell
        params = {}
        params['forget_bias'] = 0.0
        params['state_is_tuple'] = True
    elif model == 'gru':
        cell_fn = rnn.BasicGRUCell
    elif model == 'rnn':
        cell_fn = rnn.BasicRNNCell
    else:
        print('Please input correct model type!')
        exit()

    cell = cell_fn(
        num_hidden, reuse=tf.get_variable_scope().reuse,
        **params)
    cells = [cell]
    for i in range(num_layers-1):
        higher_layer_cell = cell_fn(
          num_hidden, reuse=tf.get_variable_scope().reuse,
          **params)
        cells.append(higher_layer_cell)

    cells = [tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=1.0)
               for cell in cells]

    multi_cell = rnn.MultiRNNCell(cells)
    initial_state = create_tuple_placeholders_with_default(
        multi_cell.zero_state(batch_size, tf.float32),
        extra_dims=(None,),
        shape=multi_cell.state_size)
    # initial_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, X,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, num_hidden]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [num_hidden, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    # correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    tf.summary.scalar("loss", loss)

    summ = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    return {
        'X': X,
        'seqlen': seqlen,
        'Y': Y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy,
        'summ': summ
    }

def train_graph(load_model, model, num_layers, num_hidden, g, batch_size, train_seq_length, max_epochs, weights_fpath,
    input_train_fpath, input_test_fpath, output_train_fpath, output_test_fpath):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter('log/graph', sess.graph)
        file_writer.add_graph(sess.graph)
        if load_model:
            saver.restore(sess, weights_fpath+'model_{}_{}_{}.ckpt'.format(model, num_hidden, max_epochs))
        else:
            sess.run(tf.global_variables_initializer())
        plt.ion()
        plt.figure()
        plt.show()
        mean_loss_list = []
        mean_accuracy_list = []
        current_epoch = 1
        txt_path = "results/{}_{}_{}.txt".format(model, num_hidden, max_epochs)
        best_loss = 10
        best_epoch = 1
        try:
            start_time = datetime.now()
            while current_epoch < (max_epochs + 1):
                with open(txt_path, 'a') as f:
                    print('epoch {}'.format(current_epoch), file=f)    
                seq_iter = utils.sequences(batch_size, train_seq_length, input_train_fpath, output_train_fpath)
                batch_index = -1
                total_loss = []
                total_accuracy = []
                iteration = 0
                for (input_data, y) in seq_iter:
                    batch_index += 1
                    iteration += 1
                    if input_data is not None and y is not None:
                        feed = {g['X']: input_data, g['Y']: y}
                        loss_, accuracy_, s, _ = sess.run([g['loss'], g['accuracy'], g['summ'], g['ts']], feed_dict=feed)
                        with open(txt_path, 'a') as f:
                            print('loss = %.6f' % (loss_), file=f)
                            print('accuracy = %.6f' % (accuracy_), file=f)
                        total_loss += [loss_]
                        total_accuracy += [accuracy_]
                        file_writer.add_summary(s, current_epoch)
                        print("current_epoch " + str(current_epoch) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss_) + ", Training Accuracy= " + \
                        "{:.3f}".format(accuracy_))
                mean_loss_list.append(np.mean(total_loss))
                mean_accuracy_list.append(np.mean(total_accuracy))
                if np.mean(total_loss) < best_loss:
                    best_loss = np.mean(total_loss)
                    best_epoch = current_epoch

                print("training time {}".format((datetime.now() - start_time)))
                print("  training loss:                 " + str(np.mean(total_loss)))
                print("  training accuracy:             " + str(np.mean(total_accuracy)))
                print("  best epoch:                    " + str(best_epoch))
                print("  best training loss:            " + str(best_loss))

                with open(txt_path, 'a') as f:
                    print("training time {}".format((datetime.now() - start_time)), file=f)
                    print("  training loss:                 " + str(np.mean(total_loss)), file=f)
                    print("  training accuracy:             " + str(np.mean(total_accuracy)), file=f)
                    print("  best epoch:                    " + str(best_epoch), file=f)
                    print("  best training loss:            " + str(best_loss), file=f)

                plot(mean_loss_list, mean_accuracy_list)
                plt.savefig("results/{}_{}_{}.png".format(model, num_hidden, max_epochs)) 
                current_epoch += 1
                save_path = saver.save(sess, weights_fpath+'model_{}_{}_{}.ckpt'.format(model, num_hidden, max_epochs))

            print("Optimization Finished!")
            print("Model saved in file: %s" % save_path)
        except KeyboardInterrupt:
            print('caught ctrl-c, stopping training')
            print("Model saved in file: %s" % save_path)

        print("{} epochs training time {}".format(max_epochs, datetime.now() - start_time))
        with open(txt_path, 'a') as f:
            print("{} epochs training time {}".format(max_epochs, datetime.now() - start_time), file=f)
            print("current time: {}".format(datetime.now()), file=f)


def test_graph(load_model, model, num_layers, num_hidden, g, batch_size, train_seq_length, max_epochs, weights_fpath,
    input_train_fpath, input_test_fpath, output_train_fpath, output_test_fpath):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, weights_fpath+'model_{}_{}_{}.ckpt'.format(model, num_hidden, max_epochs))
        mean_loss_list = []
        mean_accuracy_list = []
        try:  
            seq_iter = utils.sequences(batch_size, train_seq_length, input_test_fpath, output_test_fpath)
            total_loss = []
            total_accuracy = []
            for (input_data, y) in seq_iter:
                if input_data is not None and y is not None:
                    feed = {g['X']: input_data, g['Y']: y}
                    loss_, accuracy_, s, _ = sess.run([g['loss'], g['accuracy'], g['summ'], g['ts']], feed_dict=feed)
                        total_loss += [loss_]
                        total_accuracy += [accuracy_]
                        print("Minibatch Loss= " + \
                        "{:.4f}".format(loss_) + ", Testing Accuracy= " + \
                        "{:.3f}".format(accuracy_))
            
            print("Testing Loss= " + \
                        "{:.4f}".format(np.mean(total_loss)) + ", Testing Accuracy= " + \
                        "{:.3f}".format(np.mean(total_accuracy)))


def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
    if isinstance(shape, int):
        result = tf.placeholder_with_default(
        inputs, list(extra_dims) + [shape])
    else:
        subplaceholders = [create_tuple_placeholders_with_default(
        subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
        t = type(shape)
        if t == tuple:
            result = t(subplaceholders)
        else:
            result = t(*subplaceholders)    
    return result

def create_tuple_placeholders(dtype, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder(dtype, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                       for subshape in shape]
    t = type(shape)

    # Handles both tuple and LSTMStateTuple.
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result

def plot(loss_list, accuracy_list):
    plt.title('Training Loss and Accuracy')
    plt.subplot(2, 1, 1)
    plt.ylabel('loss')
    plt.plot(loss_list, 'g')

    plt.subplot(2, 1, 2)
    plt.xlabel("Epoches")
    plt.ylabel('accuracy')
    plt.plot(accuracy_list, 'r')

    plt.draw()
    plt.pause(0.0001)


