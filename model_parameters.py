#%%
import random
import cv2
import numpy as np
import shutil
import tensorflow as tf
from PIL import Image
from keras import backend as K
from art.estimators.classification import KerasClassifier
from art.metrics import empirical_robustness
import keras.backend as KTF
from keras.datasets.mnist import load_data
from keras.models import load_model
from keras.utils import to_categorical
from collections import defaultdict
from keras.models import Model
import foolbox
import matplotlib.pyplot as plt
from keras.layers import Input
import time
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False
#%%
model = load_model('/data0/jinhaibo/DGAN/train_model/cifar10_vgg.h5')
model.summary()
model_layer_dict1 = init_coverage_tables(model)
layer_names = [layer.name for layer in model.layers if
               'flatten' not in layer.name and 'input' not in layer.name]