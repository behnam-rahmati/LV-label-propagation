#!/usr/bin/env python2.7
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, Cropping2D
from keras import backend as K
from keras.losses import categorical_crossentropy as CCE
import sys
import math
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

crop_size = 100
def total_variation_loss(x):
    a = tf.square(
        x[:, : crop_size - 1, : crop_size - 1, :] - x[:, 1:, : crop_size - 1, :]
    )
    b = tf.square(
        x[:, : crop_size - 1, : crop_size - 1, :] - x[:, : crop_size - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
def custom_loss(ignore_matrix):
    def loss(ytrue, ypred):
        # print("hi")
        ignore_mat = K.cast(ignore_matrix, 'float32')
        # print("hii")

        ytrue = K.cast(ytrue, 'float32')
        # print("hiii")
        # print(np.array(ytrue).shape)
        ypred = K.cast(ypred, 'float32')
        # print("hiiiii")
        # x = K.binary_crossentropy(ytrue, ypred)
        # print(np.array(x).shape)
        # print(x)
        loss_ce = K.mean((K.binary_crossentropy(ytrue, ypred) * (1 - ignore_mat)), axis = (1,2,3))
        # print(np.array(loss_ce))
        # ypred_dx = ypred[:, :-1, :] - ypred[:, 1:, :]
        # ypred_dy = ypred[:, :, :-1] - ypred[:, :, 1:]
        # print('hi')
        y_true = np.array(ytrue).flatten()
        y_pred = np.array(ypred).flatten()
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ignore_mat = np.array(ignore_mat).flatten()
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss = loss * (1 - ignore_mat)
        loss = np.sum(loss) / np.sum(1 - ignore_mat)
        tv = 0
        # for i in range(ypred.shape[1]-1):
            # for j in range(ypred.shape[2]-1):
                # tv += np.abs(ypred[0, i+1, j, 0] - ypred[0, i, j, 0])
                # tv += np.abs(ypred[0, i, j+1, 0] - ypred[0, i, j, 0])
        loss_tv = tv
        # loss_tv = K.mean(K.abs(ypred_dx)) + K.mean(K.abs(ypred_dy))
        # print(loss_ce)
        # print(loss_tv)
        # print(np.array(loss_tv).shape)
        return loss_ce+ loss_tv
    return loss
        
    
def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn


def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.int_shape(t)
        
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h / 2, crop_h / 2 + rem_h)
    lst1=list(crop_h_dims)
    lst1[0]=int(lst1[0])
    lst1[1]=int(lst1[1])
    crop_h_dims=tuple(lst1)
    crop_w_dims = (crop_w / 2, crop_w / 2 + rem_w)
    lst2=list(crop_w_dims)
    lst2[0]=int(lst2[0])
    lst2[1]=int(lst2[1])
    crop_w_dims=tuple(lst2)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])
    
    return cropped
# def SSL (y_true , y_pred):
    # y_true_np = y_true.numpy()
    # y_pred_np = y_pred.numpy()    
    # index_reliable = np.ones ((1,100, 100,1), dtype = bool)
    # # index_reliable [(y_true_np == -1)] = False 
    # t = y_true[index_reliable]
    # p = y_pred[index_reliable]
    # t = tf.convert_to_tensor (t)
    # p = tf.convert_to_tensor (p)    
    # res= CCE(t,p)
    # #print (res)
    # return res/len (p)  
def weighted_CE (y_true , y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()    
    #y_pred_m is the y_pred with 2 classes (LV, BG)
    # y_pred_m = np.zeros (shape = (1,100,100,2), dtype = np.float32)
    # y_pred_m [:,:,:,0] = y_pred_np[:,:,:,0]  
    # y_pred_m [:,:,:,1] = 1 - y_pred_np[:,:,:,0]
    # y_pred_m = tf.convert_to_tensor(y_pred_m)
    # scale preds so that the class probas of each sample sum to 1
    # y_pred_m /= tf.reduce_sum(y_pred_m,
                            # reduction_indices=len(y_pred.get_shape()) - 1,
                            # keep_dims=True)
    # manual computation of crossentropy
    epsilon = K.epsilon()
    y_pred_m = tf.clip_by_value(y_pred_np, epsilon, 1. - epsilon)
    axes = (0,1,2)
    c_1 =  - K.sum( y_true_np[:,:,:,0] * tf.math.log(y_pred_np[:,:,:,0])) / K.sum(y_true_np [:,:,:,0])
    print(c_1)
    c_2 =  - K.sum( y_true_np[:,:,:,1] * tf.math.log(y_pred_np[:,:,:,1])) / K.sum(y_true_np [:,:,:,1])
    print(c_2)
    return c_1 + c_2
import tensorflow as tf

def masked_cross_entropy(y_true, y_pred):
    # Ignore pixels of the mask having value -1
    mask = tf.not_equal(y_true, -1)

    # Perform cross entropy only on the remaining pixels
    loss = tf.compat.v1.losses.sparse_categorical_crossentropy(y_true[mask], y_pred[mask])

    return loss
    
                               
def weak_loss (epoch):                               
    def weak_annotation (y_true , y_pred ):

        y_true_np = y_true.numpy()
        # print("this is loss")
        # print(y_pred.shape)
        y_true_1 = np.zeros((1,100,100,1), dtype = 'float32')
        y_true_2 = np.zeros((1,100,100,1), dtype = 'float32')
        y_true_1 [y_true_np == 1] = 1
        y_true_2 [y_true_np == 0] = 1
        if(K.sum((y_true_np==-1).astype(int))>0):
            SUP = 0
            # sample_weight = 0.2
        else:
            SUP = 1
            # sample_weight = 1
        # if (epoch<10) : 
            # sample_weight = SUP
        # else:
        sample_weight = (epoch/100 * (1-SUP)) + SUP
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        binary_entropy = -(K.sum((y_true_1*tf.math.log(y_pred)))/K.sum(y_true_1)) - (K.sum(y_true_2 * (tf.math.log(1- y_pred))) /K.sum(y_true_2))
        # print (binary_entropy)
        return binary_entropy * sample_weight
    return weak_annotation
# def weighted(logits, labels) :
    # '''
    # Weighted cross entropy loss, with a weight per class
    # :param logits: Network output before softmax
    # :param labels: Ground truth masks
    # :param class_weights: A list of the weights for each class
    # :return: weighted cross entropy loss
    # '''
    # class_weights = [0.1, 0.3, 0.3, 0.3]
    # n_class = len(class_weights)

    # flat_logits = tf.reshape(logits, [-1, n_class])
    # flat_labels = tf.reshape(labels, [-1, n_class])

    # class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    # weight_map = tf.multiply(flat_labels, class_weights)
    # weight_map = tf.reduce_sum(weight_map, axis=1)

    # #loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    # loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
    # weighted_loss = tf.multiply(loss_map, weight_map)

    # loss = tf.reduce_mean(weighted_loss)

    # return loss

def dice_coef(y_true, y_pred, smooth=0.0):
    # print(np.array(y_true).shape)
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)

def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)


def fcn_model(input_shape, num_classes, weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        #loss = weighted
        activation = 'sigmoid'
    else:

        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )
    
    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)
    
    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool1')(mvn3)
    print('**********pool1*****************')                
    print(pool1.shape)
    
    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool2')(mvn7)

    print('**********pool2*****************')                
    print(pool2.shape)
    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)
    print('**********pool3*****************')                
    print(pool3.shape)

    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)


    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')
    
    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')
    
    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample3')(fuse_scores2)
    print('**********upsample3*****************')                
    print(upsample3.shape)                        
    crop3 = Lambda(crop, name='crop3')([data, upsample3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=activation, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='predictions')(crop3)

    model = Model(inputs=data, outputs=predictions)
    # model.save_weights('my_initial_weights.h5')

    if weights is not None:
        model.load_weights(weights)
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)

    return model


if __name__ == '__main__':
    model = fcn_model((100, 100, 1), 2, weights=None)
