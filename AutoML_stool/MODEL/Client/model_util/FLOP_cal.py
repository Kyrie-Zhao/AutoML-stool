import numpy as np
import tensorflow as tf
import os
from VGG import B_VGGNet
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def conv_flops(k_size, c_in, c_out, h_out, w_out):
    return 2 * (k_size ** 2) * c_in * h_out * w_out * c_out

def fc_flops(num_in, num_out):
    return 2 * num_in * num_out

if __name__ == "__main__":
    img_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 3], name='image_placeholder')
    training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
    # create model and build graph
    B_VGGNet_instance = B_VGGNet(num_class=10)
    [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = B_VGGNet_instance.model(img_placeholder, is_train=training_flag)

    layers = {}

    parts = {
        'baseline_1': [['baseline/conv1', 'baseline/conv2','baseline/conv3', 'baseline/conv4'], 0],
        'baseline_2': [['baseline/conv5','baseline/conv6','baseline/conv7'], 0],
        'baseline_3': [['baseline/conv8','baseline/conv9','baseline/conv10'], 0],
        'baseline_4': [['baseline/conv11', 'baseline/conv12', 'baseline/conv13', 'baseline/fc3', 'baseline/fc4', 'baseline/logits_exit3'], 0],
        'exit0': [['exit0/fc1', 'exit0/logits_exit0'], 0],
        'exit1': [['exit1/fc1', 'exit1/logits_exit1'], 0],
        'exit2': [['exit2/fc1', 'exit2/fc2',   'exit2/logits_exit2'], 0]
    }

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables())
    for var in tf.trainable_variables():
        if 'conv' in var.name:
            w_out, h_out = sess.graph.get_tensor_by_name(var.name.replace('/kernel:0', '_relu:0')).get_shape().as_list()[1:3]
            #print(w_out, h_out)
            k_size, _, c_in, c_out = var.get_shape().as_list()
            print(k_size, c_in, c_out)
            op_name = var.name.replace('/kernel:0', '')
            layers[op_name] = conv_flops(k_size, c_in, c_out, h_out, w_out)
            #print(op_name, layers[op_name])

        if 'fc' in var.name or 'logits_exit' in var.name:
            print(var.name)

            num_in, num_out = sess.graph.get_tensor_by_name(var.name).get_shape().as_list()
            print(num_in)
            print(num_out)
            print("fuck")
            op_name = var.name.replace('/kernel:0', '')
            layers[op_name] = fc_flops(num_in, num_out)

    tmp = []
    for key in parts.keys():
        for layer in parts[key][0]:
            parts[key][1] += layers[layer]
        print("{}: {}".format(key, str(parts[key][1] / 1000000)+" MFLOPs"))
        tmp.append(parts[key][1] / 1000000)

    sess.close()

# Three Branches
    exit_0 = tmp[0]+tmp[4]
    exit_1 = tmp[0]+tmp[1]+tmp[5]
    exit_2 = tmp[0]+tmp[1]+tmp[2]+tmp[6]
    exit_3 = tmp[0]+tmp[1]+tmp[2]+tmp[3]
    print("exit 0: {}, exit 1: {}, exit2: {}, exit3: {} MFLOPS".format(exit_0,exit_1,exit_2,exit_3))
    print("real flops: exit 0: {}, exit 1: {}, exit2: {}, exit3: {} MFLOPS".format(exit_0,exit_1+tmp[4],exit_2 +tmp[4]+tmp[5],exit_3 +tmp[3]+tmp[4]+tmp[5]))
# Two Branches

"""    exit_0 = tmp[0]+tmp[3]
    exit_1 = tmp[0]+tmp[1]+tmp[4]
    exit_2 = tmp[0]+tmp[1]+tmp[2]
    print("exit 0: {}, exit 1: {}, exit2: {} MFLOPS".format(exit_0,exit_1,exit_2))
    print("real flops: exit 0: {}, exit 1: {}, exit2: {} MFLOPS".format(exit_0,exit_1+tmp[3],exit_2 +tmp[4] + tmp[3]))"""
