

import numpy as np
import nibabel as nib                                                     
import itk                                                                
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
from skimage.util import montage 
from skimage.transform import rotate
from datetime import datetime
import seaborn as sns
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
import cv2

print("\n"*5)

image_path = "./training-data/BraTS2021_00284/BraTS2021_00284_t1.nii.gz"
mask_path = "./training-data/BraTS2021_00284/BraTS2021_00284_seg.nii.gz"

image_obj = nib.load(image_path)
mask_obj = nib.load(mask_path)

image_data = image_obj.get_fdata()
mask_data = mask_obj.get_fdata()

def visualize_3d(layer):
    return layer

def visualize_3d_labels(layer):
    return layer

#plt.imshow(mask_data[:, :, 100])
#plt.imshow(image_data[:, :, 100], cmap='gray')
#plt.imshow(image_data[:, :, 105], cmap='gray')
w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 6 
for i in range(1, 12+1):
    img = image_data[:, :, 100+i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gray')
for i in range(1, 12+1):
    img = mask_data[:, :, 100+i]
    fig.add_subplot(rows, columns, i+12)
    plt.imshow(img)

#plt.show()











#### Borrowed a little from: https://www.kaggle.com/code/malik12345/brain-tumor-detection-using-cnn-model/notebook




# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    """
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))


 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[0,:,:,:,1] * y_pred[0,:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,1])) + K.sum(K.square(y_pred[0,:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[0,:,:,:,2] * y_pred[0,:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,2])) + K.sum(K.square(y_pred[0,:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[0,:,:,:,3] * y_pred[0,:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,3])) + K.sum(K.square(y_pred[0,:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())






























IMG_SIZE=image_data.shape[0] # 240
SLICES=image_data.shape[2]-3 # 155 (minus 3 = 152 s.t. we can divide by 2 three times)
BATCH_SIZE=1

def unet_3d_conv(layer, filters):
    layer = Conv3D(filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    return layer


def unet_3d(input_img):
    c1 = unet_3d_conv(input_img, 32)
    c2 = unet_3d_conv(c1, 64)
    c3 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(c2)

    c4 = unet_3d_conv(c3, 64)
    c5 = unet_3d_conv(c4, 128)
    c6 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(c5)

    c7 = unet_3d_conv(c6, 128)
    c8 = unet_3d_conv(c7, 256)
    c9 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(c8)

    c10 = unet_3d_conv(c9, 256)
    c11 = unet_3d_conv(c10, 512)
    c12 = UpSampling3D(2)(c11)

    c13 = concatenate([c8, c12])
    c14 = unet_3d_conv(c13, 256)
    c15 = unet_3d_conv(c14, 256)
    c16 = UpSampling3D(2)(c15)

    c17 = concatenate([c5, c16])
    c18 = unet_3d_conv(c17, 128)
    c19 = unet_3d_conv(c18, 128)
    c20 = UpSampling3D(2)(c19)

    c21 = concatenate([c2, c20])
    c22 = unet_3d_conv(c21, 64)
    c23 = unet_3d_conv(c22, 64)
    c24 = Conv3D(4, kernel_size=(1,1,1), strides=(1,1,1), padding='same')(c23)
    c25 = Activation('softmax')(c24)


    model = Model(inputs=input_img, outputs=c25)
    return model 


input_layer = Input((SLICES, IMG_SIZE, IMG_SIZE, BATCH_SIZE))
model = unet_3d(input_layer) 
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )
model.summary()















































