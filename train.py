import os
import random
import tensorflow as tf
from tensorflow import keras
import time
import cv2
import numpy as np

## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# SEM GPU é muito mais rápido e dá menos erro
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#data generator class
class DataGen(keras.utils.Sequence):
    def __init__(self, train_ids, train_path, batch_size=3, image_size=512):
        self.ids = train_ids;
        self.path = train_path;
        self.batch_size = batch_size;
        self.image_size = image_size;
        self.on_epoch_end();
        
    def __load__(self, id_name):
        #defining path for the training data set
        
        id_name_mask = id_name.replace(".JPG",".png").replace(".jpg",".png")
        image_path = os.path.join(self.path, "images", id_name);
        mask_path = os.path.join(self.path, "masks", id_name_mask); #mascaras estao em .png
        
        # reading the image from dataset
        ## Reading Image
        image = cv2.imread(image_path, 1); #reading image to image vaiable
        image = cv2.resize(image, (self.image_size, self.image_size));
        mask = cv2.imread(mask_path, 0); #mask image of same size with all zeros
        mask = cv2.resize(mask, (self.image_size ,self.image_size));
        mask = np.expand_dims(mask, axis=-1);
        #image normalisation
        image = image / 255.0
        mask = mask / 255.0
        return image, mask
    
    def __getitem__(self, index): #index : batch no.
        if(index+1)*self.batch_size > len(self.ids): # redining last batch size
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size] #list of 10 image ids
        
        image = [] #collection of 10 images
        mask  = [] #collection of 10 masks
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name);
            image.append(_img);
            mask.append(_mask);
            
        image = np.array(image);
        mask  = np.array(mask);
        
        return image, mask; #return array of 10 images and 10 masks
    
    #for printing the statistics of the function
    def on_epoch_end(self):
        print("epoch completed");
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)));#length of the epoch as length of generation DataGen obj


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def UNet(size):
    if (size == 128):
        return UNet128()
    elif (size == 256):
        return UNet256()
    elif (size == 512):
        return UNet512()
    elif (size == 1024):
        return UNet1024()
    
def UNet1024():
    f = [16, 32, 64, 128, 256, 512, 1024, 2048]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #pooling layer downsmaples 128 image to 512
    c2, p2 = down_block(p1, f[1]) #pooling layer downsmaples 128 image to 256
    c3, p3 = down_block(p2, f[2]) #pooling layer downsmaples 128 image to 128
    c4, p4 = down_block(p3, f[3]) #pooling layer downsmaples 128 image to 64
    c5, p5 = down_block(p4, f[4]) #pooling layer downsmaples 128 image to 32
    c6, p6 = down_block(p5, f[5]) #pooling layer downsmaples 128 image to 16
    c7, p7 = down_block(p6, f[6]) #pooling layer downsmaples 128 image to 8
    
    bn = bottleneck(p7, f[7])
    
    u1 = up_block(bn, c7, f[6]) #upsampling layer upsmaples 8 image to 16
    u2 = up_block(u1, c6, f[5]) #upsampling layer upsmaples 8 image to 32
    u3 = up_block(u2, c5, f[4]) #upsampling layer upsmaples 8 image to 64
    u4 = up_block(u3, c4, f[3]) #upsampling layer upsmaples 8 image to 128
    u5 = up_block(u4, c3, f[2]) #upsampling layer upsmaples 8 image to 256
    u6 = up_block(u5, c2, f[1]) #upsampling layer upsmaples 8 image to 512
    u7 = up_block(u6, c1, f[0]) #upsampling layer upsmaples 8 image to 512
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u7)
    model = keras.models.Model(inputs, outputs)
    return model
    

#unet model
def UNet512():
    f = [16, 32, 64, 128, 256, 512, 1024]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0], kernel_size=(15,15)) #pooling layer downsmaples 128 image to 256
    c2, p2 = down_block(p1, f[1]) #pooling layer downsmaples 128 image to 128
    c3, p3 = down_block(p2, f[2]) #pooling layer downsmaples 128 image to 64
    c4, p4 = down_block(p3, f[3]) #pooling layer downsmaples 128 image to 32
    c5, p5 = down_block(p4, f[4]) #pooling layer downsmaples 128 image to 16
    c6, p6 = down_block(p5, f[5]) #pooling layer downsmaples 128 image to 8
    
    bn = bottleneck(p6, f[6])
    
    u1 = up_block(bn, c6, f[5]) #upsampling layer upsmaples 8 image to 16
    u2 = up_block(u1, c5, f[4]) #upsampling layer upsmaples 8 image to 32
    u3 = up_block(u2, c4, f[3]) #upsampling layer upsmaples 8 image to 64
    u4 = up_block(u3, c3, f[2]) #upsampling layer upsmaples 8 image to 128
    u5 = up_block(u4, c2, f[1]) #upsampling layer upsmaples 8 image to 256
    u6 = up_block(u5, c1, f[0], kernel_size=(15,15)) #upsampling layer upsmaples 8 image to 512
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u6)
    model = keras.models.Model(inputs, outputs)
    return model

def UNet256():
    f = [16, 32, 64, 128, 256, 512]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #pooling layer downsmaples 128 image to 128
    c2, p2 = down_block(p1, f[1]) #pooling layer downsmaples 128 image to 64
    c3, p3 = down_block(p2, f[2]) #pooling layer downsmaples 128 image to 32
    c4, p4 = down_block(p3, f[3]) #pooling layer downsmaples 128 image to 16
    c5, p5 = down_block(p4, f[4]) #pooling layer downsmaples 128 image to 8
    
    bn = bottleneck(p5, f[5])
    
    u1 = up_block(bn, c5, f[4]) #upsampling layer upsmaples 8 image to 16
    u2 = up_block(u1, c4, f[3]) #upsampling layer upsmaples 8 image to 32
    u3 = up_block(u2, c3, f[2]) #upsampling layer upsmaples 8 image to 64
    u4 = up_block(u3, c2, f[1]) #upsampling layer upsmaples 8 image to 128
    u5 = up_block(u4, c1, f[0]) #upsampling layer upsmaples 8 image to 256
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)
    model = keras.models.Model(inputs, outputs)
    return model


def UNet128():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #pooling layer downsmaples 128 image to 64
    c2, p2 = down_block(p1, f[1]) #pooling layer downsmaples 128 image to 32
    c3, p3 = down_block(p2, f[2]) #pooling layer downsmaples 128 image to 16
    c4, p4 = down_block(p3, f[3]) #pooling layer downsmaples 128 image to 8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #upsampling layer upsmaples 8 image to 16
    u2 = up_block(u1, c3, f[2]) #upsampling layer upsmaples 8 image to 32
    u3 = up_block(u2, c2, f[1]) #upsampling layer upsmaples 8 image to 64
    u4 = up_block(u3, c1, f[0]) #upsampling layer upsmaples 8 image to 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model



#hyperparameter
#ATENCAO 512 trava o PC
image_size = 512
train_path = "./data-1024x1024 remade/"; #address of the dataset
batch_size = 10
epochs = 3000

#train path
train_ids = os.listdir(train_path + "/images")
val_data_size = 5; #size of set of images used for the validation 
valid_ids = train_ids[:val_data_size]; # list of image ids used for validation of result 0 to 9
train_ids = train_ids[val_data_size:]; #list of image ids used for training dataset
gen = DataGen(train_ids, train_path, batch_size, image_size)

model = UNet(image_size)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

print("training_size: ", len(train_ids), "validation_size: ", len(valid_ids))
print("total epoch: ", len(gen))
print("total training batches: ", len(train_gen))
print("total validaton batches: ", len(valid_gen))
print("image_size:", image_size)
print(epochs)
versao = 1

_fileName = "/cells-s%d-e%d-v%d-tf241.h5"
filename = _fileName % (image_size, epochs, versao)

print("filename:",filename)

ini = time.time()

history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, 
                        validation_steps=valid_steps, epochs=epochs);
filename = _fileName % (image_size, epochs, versao)

fim = time.time()

print("Tempo:", fim-ini)
print("Horas:", (fim-ini)/60/60)
print(filename)
model.save(filename)