#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from keras.layers import Input
from keras.datasets.cifar10 import load_data
from keras.models import load_model
import cv2
import time
import math
# from utils import *
import numpy as np
import tensorflow as tf
# from Model_Load import Model_load
from keras.utils import to_categorical
from keras.models import load_model
import keras.backend as KTF
import matplotlib.pyplot as plt
import foolbox
import os
import argparse
from keras import backend as K
from keras.models import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.compat.v1.disable_eager_execution()


#%%
parser = argparse.ArgumentParser(description='input flags')
parser.add_argument('--flags', type=int,
                    help='running loop')
args = parser.parse_args()
#%%
def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def coverage_value(input_data, model):
    layer_names = [layer.name for layer in model.layers if 'fc2' in layer.name]
    get_value = [[] for j in range(len(layer_names))]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        # scaled = scale(intermediate_layer_output)
        for num_neuron in range(intermediate_layer_output.shape[-1]):
            get_value[i].append(np.mean(intermediate_layer_output[..., num_neuron]))
    return get_value

def update_coverage_value(input_data, model, layers):
    layer_names = layers
    get_value = [[] for j in range(len(layer_names))]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_names).output])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output)
        for num_neuron in range(scaled.shape[-1]):
            get_value[i].append(np.mean(scaled[..., num_neuron]))
    return get_value

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def compute_tk(path, k, layer_names, l):
    temp = []
    dict = {}
    neuron_index = []
    for i in range(len(path)):
        for j in range(k * (l+1)):
            if path[i][j][0][0] == layer_names:
                temp.append(path[i][j][0][1])
    for key in temp:
        dict[key] = dict.get(key, 0) + 1
    neuron_name = []
    neuron_value = []
    print(dict)
    for key in dict:
        neuron_value.append(dict[key])
        neuron_name.append(key)
    neuron_value = np.array(neuron_value)
    value = neuron_value.argsort()[::-1]
    value = value[:k]
    for v in range(k):
        # print(k)
        neuron_index.append(neuron_name[value[v]])
    # print(neuron_index)
    return dict, neuron_index


def build_all_loss(model, seeds, l, k):
    path = []
    layers = [layer.name for layer in model.layers if l in layer.name]
    for i in range(len(seeds)):
        path_temp = []
        x = seeds[i:i+1]
        neuron_value = update_coverage_value(x, model, layers)
        for m in range(len(layers)):
            neuron_value[m] = np.array(neuron_value[m])
            topk = neuron_value[m].argsort()[::-1]
            for j in range(k):
                topk_neurons = [(layers[m], topk[j], neuron_value[m][topk[j]])]
                path_temp.append(topk_neurons)
        path.append(path_temp)
        for t in range(len(layers)):
            dict, index = compute_tk(path, k, layers[t], t)
    return index

#%%fine-tuning
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# x_test = x_test
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)
K.clear_session()
model = load_model('/data0/jinhaibo/DGAN/train_model/cifar10_vgg.h5')
advs = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/PGD/adv.npy')
benigns = x_train[:5000]
#%%
count = 0
tpr = 0
print('in', args.flags)
flags = args.flags
perturbation_size = 0.001
tmp = []
for index in range(50):
    x = benigns[index + flags * 50:index+1 + flags * 50]
    predcitions = np.argmax(model.predict(x))
    output_label = model.output[:,predcitions]
    layers_names = 'flatten'
    grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.get_layer(layers_names).output])
    weights, out_value = iterate([x])
    fc_values = out_value

    for i in range(len(weights)):
        fc_values[:, i] *= weights[i]

    k = 1
    tale = 5
    out_sort = np.argsort(fc_values[0])[::-1][:k]
    tale_out_sort = np.argsort(fc_values[0])[:tale]
    loss1 = -1 * K.mean(model.layers[-1].output[..., predcitions])
    for i in range(len(out_sort)):
        if i == 0:
            loss_neuron = K.mean(model.get_layer(layers_names).output[..., out_sort[0]])
        else:
            loss_neuron = loss_neuron + K.mean(model.get_layer(layers_names).output[..., out_sort[i]])

    for i in range(len(tale_out_sort)):
        if i == 0:
            tale_neuron = K.mean(model.get_layer(layers_names).output[..., tale_out_sort[0]])
        else:
            tale_neuron = tale_neuron + K.mean(model.get_layer(layers_names).output[..., tale_out_sort[i]])

    layer_output = 1 * loss_neuron - 1 * tale_neuron
    final_loss = K.mean(layer_output)
    grads = normalize(K.gradients(final_loss, model.input)[0])
    iterate = K.function([model.input],
                         [layer_output, grads])
    # print(np.argmax(model.predict(x)))
    x_adv = advs[index + flags * 50:index+1 + flags * 50]
    if math.isnan(np.max(x_adv)):
        continue
    # elif np.linalg.norm(x_adv - x) > 6:
    #     continue
    else:
        tpr += 1

        # print(np.argmax(model.predict(x_adv)))
        print('running index', index)
        layer_output, grads_value = iterate([x])
        dict = {
            "index": index + flags * 50,
            "peturbation": grads_value

        }
        tmp.append(dict)
        # array_p = cv2.cvtColor(grads_value.reshape(32, 32, 3).astype('float32'), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/data0/jinhaibo/DGAN/Inverse_Peturbation/defense_cifar/FGSM/perturbation/perturbation_' + str(index + flags * 50) + '.png',
        #             array_p)
        x_defense = x_adv + perturbation_size * grads_value

        if np.argmax(model.predict(x_defense)) != predcitions:
            x_defense = x_adv - perturbation_size * grads_value
        x_defense = np.clip(x_defense, a_max=1, a_min=0)
        # array = cv2.cvtColor(x_defense.reshape(32, 32, 3).astype('float32'), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/data0/jinhaibo/DGAN/Inverse_Peturbation/defense_cifar/FGSM/pic/defense_' + str(index + flags * 50) + '.png', array)

        if np.argmax(model.predict(x_defense)) == predcitions:
            count += 1
        else:
            print('fail index', index)
print(count/tpr)
#%%
np.save('/data0/jinhaibo/DGAN/Inverse_Peturbation/diff_k/k1_tmp/peturbation_dict_' + str(args.flags) + '.npy', tmp)
#%%
# bb = list(np.load('/data0/jinhaibo/DGAN/Inverse_Peturbation/peturbation_dict.npy', allow_pickle=True))
# #%%
# f1 = open('/data0/jinhaibo/DGAN/Inverse_Peturbation/generation3.txt', 'w')
# for i in range(80, 100):
#     f1.write('python Reverse_VGG_CIFAR3.py --flags ' + str(i) +'\n')
#     print(i)
# f1.close()
#%%
# #%%
# i = 26
# plt.imshow(advs[500 + i:500 + i + 1].reshape(32, 32, 3))
# plt.show()
# #%%
# index = 14
# x = benigns[index + 500:index + 1 + 500]
# predcitions = np.argmax(model.predict(x))
# output_label = model.output[:predcitions]
# layers_names = 'flatten'
# grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
# out_grads = K.mean(grads, axis=0)
# iterate = K.function([model.input],
#                      [out_grads, model.get_layer(layers_names).output])
# weights, out_value = iterate([x])
# fc_values = out_value
#
# for i in range(len(weights)):
#     fc_values[:, i] *= weights[i]
#
# k = 5
# tale = 5
# out_sort = np.argsort(fc_values[0])[::-1][:k]
# tale_out_sort = np.argsort(fc_values[0])[k:k + tale]
# loss1 = -1 * K.mean(model.layers[-1].output[..., predcitions])
# for i in range(len(out_sort)):
#     if i == 0:
#         loss_neuron = K.mean(model.get_layer(layers_names).output[..., out_sort[0]])
#     else:
#         loss_neuron = loss_neuron + K.mean(model.get_layer(layers_names).output[..., out_sort[i]])
#
# for i in range(len(tale_out_sort)):
#     if i == 0:
#         tale_neuron = K.mean(model.get_layer(layers_names).output[..., tale_out_sort[0]])
#     else:
#         tale_neuron = tale_neuron + K.mean(model.get_layer(layers_names).output[..., tale_out_sort[i]])
#
# layer_output = loss1 + 1 * loss_neuron - 1 * tale_neuron
# final_loss = K.mean(layer_output)
# grads = normalize(K.gradients(final_loss, model.input)[0])
# iterate = K.function([model.input],
#                      [layer_output, grads])
# # print(np.argmax(model.predict(x)))
# x_adv = advs[index + 500:index + 1 + 500]
#
# tpr += 1
# perturbation_size = 0.005
# print(np.argmax(model.predict(x)))
# print(np.argmax(model.predict(x_adv)))
# print('running index', index)
# layer_output, grads_value = iterate([x])
# x_defense = x_adv + perturbation_size * grads_value
# if np.argmax(model.predict(x_defense)) != predcitions:
#     x_defense = x_adv - perturbation_size * grads_value
# x_defense = np.clip(x_defense, a_max=1, a_min=0)
# print(np.argmax(model.predict(x_defense)))
# plt.imshow(x_adv.reshape(32, 32, 3))
# plt.show()
# plt.imshow(x_defense.reshape(32, 32, 3))
# plt.show()
# if np.argmax(model.predict(x_defense)) == predcitions:
#     count += 1
# else:
#     print('fail index', index)