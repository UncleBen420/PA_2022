## This script is used to train an autoencoder on a folder of image
# it's return 2 models: the full autoencoder and only the encoder part
# it take 4 arguments:
# --model: the name of the model to train: mobil, basic, gray
# --path_in: the path to the image dataset
# --epoch: number of epochs to train
# --nb_image: number of image taken from the dataset
# Exemple of command line: python train_CNN.py --model mobil --path_in PreprocessImage/ --epoch 10 --nb_image=3500


import os
import cv2
import argparse
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
 
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D,Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import MobileNetV3Small

from keras.models import Sequential
from keras import Input, Model

def model_MobilNet():
	model=Sequential()

	IMAGE_HEIGHT , IMAGE_WIDTH = 160, 160

	model.add(Input(shape=(160,160,3)))

	base_model = MobileNetV3Small(weights='imagenet', input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 3),include_top=False, include_preprocessing=False, minimalistic=True)
	print(len(base_model.layers))
	#print(base_model.summary())
	for layer in base_model.layers[:90]:
		layer.trainable = False

	model.add(base_model)

	model.add(Conv2DTranspose(filters=128,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))
	model.add(Conv2DTranspose(filters=64,kernel_size=(4,4),strides=(2,2),padding='valid',activation='relu'))
	model.add(Dropout(.2))
	model.add(Conv2DTranspose(filters=64,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.add(Conv2DTranspose(filters=32,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.add(Dropout(.2))
	model.add(Conv2DTranspose(filters=3 ,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.compile(optimizer='adam',loss='mse',metrics=['mse'])
	model.summary()
	return model
	
def model_MobilNet224():
	model=Sequential()

	IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

	model.add(Input(shape=(224,224,3)))

	base_model = MobileNetV3Small(weights='imagenet', input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 3),include_top=False, include_preprocessing=False, minimalistic=True)
	print(len(base_model.layers))
	#print(base_model.summary())
	for layer in base_model.layers[:80]:
		layer.trainable = False

	model.add(base_model)

	model.add(Conv2DTranspose(filters=128,kernel_size=(7,7),strides=(1,1),padding='valid',activation='relu'))
	model.add(Conv2DTranspose(filters=64,kernel_size=(4,4),strides=(2,2),padding='valid',activation='relu'))
	model.add(Dropout(.2))
	model.add(Conv2DTranspose(filters=64,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.add(Conv2DTranspose(filters=32,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.add(Dropout(.2))
	model.add(Conv2DTranspose(filters=3 ,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	model.compile(optimizer='adam',loss='mse',metrics=['mse'])
	model.summary()
	return model
	
def model_Basic():
	model=Sequential()
	encoder=Sequential()
	decoder=Sequential()

	IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

	model.add(Input(shape=(224,224,3)))

	encoder.add(Conv2D(filters=4 ,kernel_size=(3,3),padding='valid',input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 3),activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=8,kernel_size=(3,3),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2DTranspose(filters=16,kernel_size=(5,5),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=16,kernel_size=(7,7),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2D(filters=32,kernel_size=(9,9),strides=(1,1),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))

	decoder.add(Conv2DTranspose(filters=32,kernel_size=(7,7),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=8,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=3 ,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))

	model.add(encoder)
	model.add(decoder)


	model.compile(optimizer='adam',loss='mse',metrics=['mse'])
	model.summary()
	return model
		
def model_Basic2():
	model=Sequential()
	encoder=Sequential()
	decoder=Sequential()

	IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

	model.add(Input(shape=(224,224,3)))

	encoder.add(Conv2D(filters=32 ,kernel_size=(3,3),padding='valid',input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 3),activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=32,kernel_size=(3,3),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=8,kernel_size=(9,9),strides=(1,1),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2D(filters=8,kernel_size=(17,17),strides=(1,1),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=4,kernel_size=(33,33),strides=(1,1),padding='valid',activation='relu'))
	

	decoder.add(Conv2DTranspose(filters=4,kernel_size=(17,17),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=8,kernel_size=(9,9),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
	# FINE EXPENTION LAYER
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=3,kernel_size=(2,2),strides=(1,1),padding='valid',activation='relu'))

	model.add(encoder)
	encoder.summary()
	model.add(decoder)


	model.compile(optimizer='adam',loss='mse',metrics=['mse'])
	model.summary()
	return model
	
def model_Gray():
	model=Sequential()
	encoder=Sequential()
	decoder=Sequential()

	IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

	model.add(Input(shape=(IMAGE_HEIGHT,IMAGE_WIDTH,1)))

	encoder.add(Conv2D(filters=32 ,kernel_size=(3,3),padding='valid',input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 1),activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=8,kernel_size=(3,3),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2DTranspose(filters=16,kernel_size=(5,5),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))
	encoder.add(Conv2D(filters=16,kernel_size=(7,7),padding='valid',activation='relu'))
	encoder.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
	encoder.add(Conv2D(filters=32,kernel_size=(9,9),strides=(1,1),padding='valid',activation='relu'))
	encoder.add(Dropout(.2))

	decoder.add(Conv2DTranspose(filters=32,kernel_size=(7,7),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
	decoder.add(Conv2DTranspose(filters=8,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))
	decoder.add(Dropout(.2))
	decoder.add(Conv2DTranspose(filters=1 ,kernel_size=(2,2),strides=(2,2),padding='valid',activation='relu'))

	model.add(encoder)
	model.add(decoder)


	model.compile(optimizer='adam',loss='mse',metrics=['mse'])
	model.summary()
	return model
	
# MobileNet normalize between 1 and -1 so we have to use the following	
def mobilNormalisation(img):
	return (img / 127.5) - 1

# Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1 /!| old version
def standardNormalisation(img):
	return img / 255.
	
def trainMobile(model,epoch,train):

	tf.keras.backend.clear_session()
	epochs = epoch
	batch_size = 10

	history =  model.fit(frames_list, frames_list, epochs=epochs, validation_split=0.2, batch_size=batch_size)
	
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.show()
	model.layers[0].save(args.model+'.h5')
	model.save(args.model+'_autoencoder.h5')
	
def trainBasic(model,epoch,train):

	tf.keras.backend.clear_session()
	epochs = epoch
	batch_size = 10

	history =  model.fit(frames_list, frames_list, epochs=epochs, validation_split=0.2, batch_size=batch_size)
	
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.show()
	model.layers[0].save(args.model+'.h5')
	model.save(args.model+'_autoencoder.h5')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Split video into classes by respect of the annotation file')
	parser.add_argument('--model', required=True, help='the path to the videos')
	parser.add_argument('--path_in', metavar='path', required=True, help='the path to the images')
	parser.add_argument('--epoch', required=True, help='the path to the images')
	parser.add_argument('--nb_image', required=True, help='the path to the images')

	args = parser.parse_args()

	modelSelection = {
	 "mobil":{"Getter": model_MobilNet,"norm":mobilNormalisation, "train":trainMobile, "channels":3},
	 "basic":{"Getter": model_Basic,"norm":standardNormalisation, "train":trainBasic, "channels":3},
	 "gray":{"Getter": model_Gray,"norm":standardNormalisation, "train":trainBasic, "channels":1}
	}

	# check available gpu
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	model = modelSelection[args.model]["Getter"]()
	norm = 	modelSelection[args.model]["norm"]
	DATASET_DIR = "PreprocessImage160"
	NB_IMAGE = int(args.nb_image)


	frames_list = np.zeros([NB_IMAGE,160,160,modelSelection[args.model]["channels"]])
	files_list = os.listdir(DATASET_DIR)
	random.shuffle(frames_list)
	files_list = files_list[:NB_IMAGE]
	print("______________________________________________")
	print("Acquiring image")
	print("______________________________________________")
	counter = 0
	for file_name in files_list:
		
		
		path = os.path.join(DATASET_DIR, file_name)
		if modelSelection[args.model]["channels"] > 1:
			img = norm(cv2.imread(path, cv2.IMREAD_COLOR))
		else:
			img = norm(cv2.imread(path, 0)).reshape([224,224,1])

		normalized_img= norm(img)

		# Append the normalized frame into the frames list
		frames_list[counter] = normalized_img
		counter += 1
		count = "image " + str(counter) + "/" + str(NB_IMAGE)
		print("\r",count)


	#FEEDING THE MODEL
	print("______________________________________________")
	print("Training the model")
	print("______________________________________________")
	norm = 	modelSelection[args.model]["train"](model, int(args.epoch), frames_list)


