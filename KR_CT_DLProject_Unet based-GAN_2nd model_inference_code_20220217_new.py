# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:42:04 2021

@author: User
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# example of defining a composite model for training the generator model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation, Add
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras.layers import Dense, Flatten, Add, PReLU, Lambda, UpSampling2D  
from keras.callbacks import ModelCheckpoint

# from skimage import metrics
# from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

import pydicom
from os import listdir
from os.path import isfile, join
import os

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(1024, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
    
# 	d = Conv2D(1024, (4,4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
# 	d = BatchNormalization()(d)
# 	d = LeakyReLU(alpha=0.2)(d)
    
# 	d = Conv2D(1024, (4,4), padding='same', kernel_initializer=init)(d)
# 	d = BatchNormalization()(d)
# 	d = LeakyReLU(alpha=0.2)(d)
    
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
 
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g
 
def Upsample_Block(x_in, init):
    x = Conv2D(filters = 128, kernel_size=4, padding='same', kernel_initializer = init)(x_in)
    x = SubpixelConv2D(2)(x)
    x = Conv2D(filters = 1, kernel_size=4, padding='same')(x)
    return PReLU()(x)
      
def SubpixelConv2D(scale):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale))    



# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 1024)
	e7 = define_encoder_block(e6, 1024)
	# bottleneck, no batch norm and relu
	b = Conv2D(1024, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 1024)
	d2 = decoder_block(d1, e6, 1024)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Upsample_Block(d7, init)
    
    #g = Conv2DTranspose(1, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model
 
 
# define image shape
image_shape = (256, 256, 1)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
d_model.summary()
g_model.summary()
gan_model.summary()
# plot the model
plot_model(gan_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(g_model, to_file='g_model_plot.png', show_shapes=True, show_layer_names=True)


def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = np.random.randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y



# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=30, n_batch=10, n_patch=32):
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs

    # manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        
        
# load image data
#load dataset
base_path = 'D:\\dicom_data\\DL_DATA\\CT70to50\\dataset'
x_train = np.load(base_path + '\\x_train.npy')
y_train = np.load(base_path + '\\y_train_norm.npy')
x_val = np.load(base_path + '\\x_val.npy')
y_val = np.load(base_path + '\\y_val_norm.npy')

'''
from sklearn.preprocessing import minmax_scale
itr = 0
y_train_scaled = []
for itr in range(len(y_train)):
    y_train_scaled.append(minmax_scale(y_train[itr, :, :, 0], axis=0, copy=True))
    print(itr)

y_train_sc = np.array(y_train_scaled)
y_train_sc = y_train_sc[:, :, :, np.newaxis] 
'''
dataset = [x_train, y_train]
# train model
train(d_model, g_model, gan_model, dataset)

d_model.save_weights(base_path + "\\d_model_220126_32patch.h5")
g_model.save_weights(base_path + "\\g_model_220126_32patch.h5")
gan_model.save_weights(base_path + "\\gan_model_220126_32patch.h5")

g_model.load_weights(base_path + '\\g_model_220126_32patch.h5')


img_num = 160

savepic = os.path.join('D:\\dicom_data\\DL_DATA\\CT70to50\\result_png\\' + str(img_num))
if not(os.path.exists(savepic)):
    os.makedirs(savepic)

pre_img = x_val[img_num, :, :, :]
ccc = pre_img[np.newaxis, :, : ,:]
gen_img = g_model.predict(ccc) * 65535

gen_img2 = g_model.predict(gen_img) * 65535
gen_img_subt = (y_val[img_num, :, :, 0] * 65535) - gen_img[0, :, :, 0]

plt.figure()
plt.title("input")               # nrows=2, ncols=1, index=1
plt.imshow(x_val[img_num, :, :, :], cmap = 'gray')
plt.savefig(savepic + "/input.png")

plt.figure()
plt.title("generated")
plt.imshow(gen_img[0, :, :, :], cmap = 'gray')
plt.savefig(savepic + "/generated.png")

plt.figure()
plt.title("twice")
plt.imshow(gen_img2[0, :, :, :], cmap = 'gray')
plt.savefig(savepic + "/twice.png")

plt.figure()
plt.title("difference")
plt.imshow(gen_img_subt, cmap = 'gray')
plt.savefig(savepic + "/difference.png")

plt.figure()
plt.title("Ground Truth")
plt.imshow(y_val[img_num, :, :, :], cmap = 'gray')
plt.savefig(savepic + "/GT.png")

img_x = x_train[img_num, :, :, :]
img_gen = gen_img[0, :, :, :]
img_gt = y_train[img_num, :, :, :]



# MSE = metrics.mean_squared_error(img_gen, img_gt)
# RMSE = sqrt(MSE)
# PSNR = metrics.peak_signal_noise_ratio(img_gen, img_gt)
# SSIM = metrics.structural_similarity(img_gen, img_gt, multichannel=True)



# print ('PSNR', PSNR)
# print ('SSIM', SSIM)
# print ('MSE:', MSE)
# print ('RMSE', RMSE)

#%%

import pydicom
from os import listdir
from os.path import isfile, join
import os
import cv2
import SimpleITK as sitk
import pydicom._storage_sopclass_uids

images_path = 'D:\\dicom_data\\DL_DATA\\CT70to50'

savedir2 = os.path.join(images_path +'\\AI_Mono')
if not(os.path.exists(savedir2)):
    os.makedirs(savedir2)
    
savedir3 = os.path.join(images_path +'\\70kev_256')
if not(os.path.exists(savedir3)):
    os.makedirs(savedir3)
    
savedir4 = os.path.join(images_path +'\\50kev_256')
if not(os.path.exists(savedir4)):
    os.makedirs(savedir4)

# savedir_tmp = os.path.join(images_path +'\\TEMP')
# if not(os.path.exists(savedir_tmp)):
#     os.makedirs(savedir_tmp)

path_tmp = []
name_tmp = []


for (path, dir, files) in os.walk(images_path + '\\70kev'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        
        if ext == '.dcm' or '.IMA':
            print("%s/%s" % (path, filename))
            path_tmp.append(path)
            name_tmp.append(filename)


# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

img_tmp = []
dcm_tmp = []
print("파일 로딩 중 입니다~ 처리 데이터 양이 많을 수록 오래 기다려주셔야 합니다 ㅠㅠ")
for i in range(len(path_tmp)):
    dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
    dcm_tmp.append(dcm_p)
    ccc = dcm_p.pixel_array
    ccc_resize = cv2.resize(ccc, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    dcm_p.file_meta = fileMeta
     
    dcm_p.Rows = ccc_resize.shape[0]
    dcm_p.Columns = ccc_resize.shape[1]
    #dcm_p.NumberOfFrames = ccc_resize.shape[2]
    
    #dcm_p.BitsAllocated = 16
    dcm_p.PixelRepresentation = 1
    dcm_p.PixelData = ccc_resize.tobytes()
    dcm_p.save_as(savedir3 + '/70kev_' + str(name_tmp[i]), write_like_original=False)
    
    
    gen_img = np.uint16(g_model.predict(ccc_resize[np.newaxis, :, :, np.newaxis]) * 65535)
    gen_array = gen_img[0, :, :, 0]
    dcm_p.PixelData = gen_array.tobytes()
    dcm_p.save_as(savedir2 + '/AIMono_' + str(name_tmp[i]), write_like_original=False)
    
    dcm_y = pydicom.dcmread(images_path + '/50kev/' + name_tmp[i], force = True)
    yyy = dcm_y.pixel_array
    yyy_resize = cv2.resize(yyy, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    dcm_y.file_meta = fileMeta
     
    dcm_y.Rows = yyy_resize.shape[0]
    dcm_y.Columns = yyy_resize.shape[1]
    #dcm_p.NumberOfFrames = ccc_resize.shape[2]
    #dcm_p.BitsAllocated = 16
    dcm_y.PixelRepresentation = 1
    dcm_y.PixelData = yyy_resize.tobytes()
    dcm_y.save_as(savedir4 + '/50kev_' + str(name_tmp[i]), write_like_original=False)
    
    
    #dcm_y.PixelData = yyy_resize[:, :]
    #dcm_y.save_as(savedir4 + '/' + str(name_tmp[i]) + ".dcm")
    print(i)
#%%
'''
img_tmp = []
dcm_tmp = []
print("파일 로딩 중 입니다~ 처리 데이터 양이 많을 수록 오래 기다려주셔야 합니다 ㅠㅠ")
for i in range(len(path_tmp)):
    dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
    dcm_tmp.append(dcm_p)
    ccc = dcm_p.pixel_array
    ccc_resize = cv2.resize(ccc, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    #img_tmp.append(ccc_resize)
    img_array = sitk.GetImageFromArray(ccc_resize)
    sitk.WriteImage(img_array, savedir3 + '/70kev_' + str(name_tmp[i]))
    
    #dcm_x = pydicom.dcmread(savedir_tmp + '/' + name_tmp[i], force = True)
    
    #dcm_p.PixelData = dcm_x.PixelData
    #dcm_p.Pixelarray = ccc_resize[:, :]
    #dcm_p.save_as(savedir3 + '/' + str(name_tmp[i]))
    
    gen_img = np.uint16(g_model.predict(ccc_resize[np.newaxis, :, :, np.newaxis]) * 65535)
    gen_array = sitk.GetImageFromArray(gen_img[0, :, :, 0])
    sitk.WriteImage(gen_array, savedir2 + '/AIMono_' + str(name_tmp[i]))
    #dcm_p.PixelData = gen_img[0, :, :, 0]
    #dcm_p.save_as(savedir2 + '/' + str(name_tmp[i]) + ".dcm")
    
    dcm_y = pydicom.dcmread(images_path + '/50kev/' + name_tmp[i], force = True)
    yyy = dcm_y.pixel_array
    yyy_resize = cv2.resize(yyy, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    y_img_array = sitk.GetImageFromArray(yyy_resize)
    sitk.WriteImage(y_img_array, savedir4 + '/50kev_' + str(name_tmp[i]))
    
    #dcm_y.PixelData = yyy_resize[:, :]
    #dcm_y.save_as(savedir4 + '/' + str(name_tmp[i]) + ".dcm")
    print(i)
'''

#%%   
#!/usr/bin/python3
import numpy
import pydicom
import pydicom._storage_sopclass_uids

# dummy image
image = numpy.random.randint(2**16, size=(512, 512, 512), dtype=numpy.uint16)

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# dataset
ds = pydicom.Dataset()
ds.file_meta = fileMeta

ds.Rows = image.shape[0]
ds.Columns = image.shape[1]
ds.NumberOfFrames = image.shape[2]

ds.PixelSpacing = [1, 1] # in mm
ds.SliceThickness = 1 # in mm

ds.BitsAllocated = 16
ds.PixelRepresentation = 1
ds.PixelData = image.tobytes()

# save
ds.save_as('image.dcm', write_like_original=False)    
    
#zzz = dcm_tmp[0]
#plt.hist(zzz.pixel_array)
    

# # img_num = 10

# # pre_img = img_tmp[img_num]
# # ccc = pre_img[np.newaxis, :, : ,np.newaxis]
# # gen_img = g_model.predict(ccc)

# # dcm_tmp[img_num].pixel_array = gen_img[0, :, :, :]


        
# # dcm_tmp[img_num].save_as(savedir2 + '/' + str(dcm_tmp.SeriesInstanceUID) + str(img_num) + ".dcm")
