import tensorflow as tf
import numpy as np

def cross_entropy(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, 
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy_, name='cross_entropy')
    return cross_entropy_mean

def class_balanced_cross_entropy_loss(logits, labels, samples_per_cls, beta=0.9999):
    num_classes = len(samples_per_cls)
    one_hot_labels = tf.one_hot(labels, num_classes)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
   
    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    weights = tf.tile(weights, [tf.shape(one_hot_labels)[0], 1]) * one_hot_labels
    weights = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, num_classes])

    tower_loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits, 
                                                 weights=tf.reduce_mean(weights, axis=1))
    tower_loss = tf.reduce_mean(tower_loss)
    return tower_loss

def top_k_error(predictions, labels, k=1):
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return num_correct / float(batch_size)