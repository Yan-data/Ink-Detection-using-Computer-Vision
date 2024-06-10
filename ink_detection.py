# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:00:07 2024

@author: Yan
"""
from tensorflow import keras
import albumentations as A
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator
import cv2
#from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import tensorflow as tf
from pathlib import Path



####------------------------------------------------------------------

    
# ============== comp exp name =============
Z_START =30
Z_DIMS = 16
tile_size = 256
OUTPUT_DIR = Path( f"data/crops/crop{tile_size}_{Z_START}_{Z_DIMS}")
comp_dataset_path = OUTPUT_DIR


# ============== augmentation =============
train_aug_list = [
    #A.Resize(size, size),
    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.75),
    A.OneOf([
        A.GaussNoise(var_limit=[10, 50]),
        A.GaussianBlur(),
        A.MotionBlur(),
    ], p=0.4),
    #A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    
]
  
# Read data

#fold = [1]

img_fns= glob.glob(f'{comp_dataset_path}/*.npy')
img_fns = sorted(img_fns)
masks_fns= glob.glob(f'{comp_dataset_path}/*.png')
masks_fns = sorted(masks_fns)
print('size of X and y:', len(img_fns), len(masks_fns))

#%
#img_fns, masks_fns=get_fns(mode='valid')
x_train_list = []
y_train_list= []
#train_index=[0:150]
for path_img, path_mask in zip(img_fns[0:200],masks_fns[0:200]):
    x_train_list.append(np.load(path_img).astype(int))
    mask=cv2.imread(path_mask).astype(int)#(256,256,3)
    m1=mask[:,:,[0]]
    m1[m1>0]=1
    y_train_list.append(m1)
    

x_train=np.array(x_train_list)
y_train=np.array(y_train_list)

#%
#validation data
x_val_list = []
y_val_list = []
#val_index=[150:180]
for path_img, path_mask in zip(img_fns[200:250],masks_fns[200:250]):

    x_val_list.append(np.load(path_img).astype(int))
    mask=cv2.imread(path_mask).astype(int)
    m1=mask[:,:,[0]]
    m1[m1>0]=1
    y_val_list.append(m1)

x_val=np.array(x_val_list)
y_val=np.array(y_val_list)
    
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

#%%

trans_train = A.Compose(train_aug_list)

 
def aug_fn_train(image):
    data = {"image":image}
    aug_data = trans_train(**data)
    aug_img = aug_data["image"]
    #aug_img = tf.cast(image/65535.0, tf.float32)
    #aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img

def augmen(image):
    #augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    aug_img = tf.numpy_function(aug_fn_train, [image], tf.float32)
    return aug_img

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # augment volume
    #volume = augmen(volume)
    #volume = volume/65535.0
    volume=tf.cast(volume, tf.float32)
    label=tf.cast(label, tf.float32)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    #volume = volume/65535.0
    volume=tf.cast(volume, tf.float32)
    label=tf.cast(label, tf.float32)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


#%%

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

#%%
#Visualize an augmented CT scan.
data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the X ray scan slice is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 0]), cmap="gray")

#%%
# model
#from tensorflow import keras
#import tensorflow as tf

#    """Build a 3D convolutional neural network model."""

#inputs = keras.layers.Input((width, height, depth, 1))
input_ = keras.layers.Input((128, 128, 16, 1))

out1 = keras.layers.Conv3D(3, 3, padding='same',
                       activation='relu')(input_)  # 
out2 = keras.layers.MaxPool3D()(out1)  # 40, 40, 4
nb1=tf.keras.layers.BatchNormalization()(out2)
out3 = keras.layers.Conv3D(6, 3, padding='same',
                       activation='relu')(nb1)  # 
out4 = keras.layers.MaxPool3D()(out3)  # 20, 20, 12
nb2=tf.keras.layers.BatchNormalization()(out4)
out5 = keras.layers.Conv3D(12, 3, padding='same',
                       activation='relu')(nb2)  # 
out6 = keras.layers.MaxPool3D()(out5)  # 10, 10, 24
nb3=tf.keras.layers.BatchNormalization()(out6)
out7 = keras.layers.Conv3D(24, 3, padding='same',
                       activation='relu')(nb3)  # 
out8 = keras.layers.MaxPool3D()(out7)  # 
nb4=tf.keras.layers.BatchNormalization()(out8)

out9 = keras.layers.Conv3DTranspose(
12, 3, padding='same', activation='relu', strides=2)(nb4)
out10 = keras.layers.Add()((out9, out6)) #

out11 = keras.layers.Conv3DTranspose(
6, 3, padding='same', activation='relu', strides=2)(out10)
out12 = keras.layers.Add()((out11, out4)) #


out13 = keras.layers.Conv3DTranspose(
3, 3, padding='same', activation='relu', strides=2)(out12)
out14 = keras.layers.Add()((out13, out2)) # 


out15 = keras.layers.Conv3DTranspose(
3, 3, padding='same', activation='relu', strides=2)(out14)
out16 = keras.layers.Concatenate()((out15, input_)) #

###

out17 = keras.layers.Lambda(lambda x:tf.reduce_max(x, axis=3))(out16) #
out18 = keras.layers.Conv2D(2, 3, padding='same',activation='relu')(out17)  
#out19 = keras.layers.Lambda(lambda x:tf.reduce_mean(x, axis=3))(out18)  
out19=keras.layers.Flatten()(out18) 
outputs = keras.layers.Dense(1, activation='sigmoid')(out19)
#outputs =keras.layers.Reshape((128,128))(out19)
 

# Define the model.
model = keras.models.Model(input_, outputs,name='3d_unet')


# Build model.
model.summary()


#%%


recall = keras.metrics.Recall()
precision = keras.metrics.Precision()
#fb=keras.metrics.FBetaScore(beta=0.5,threshold=0.5)
#from sklearn.metrics import fbeta_score

# def fbeta_numpy(targets, preds, beta=0.5):
#     """
#     https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
#     """
#     y_true_count = tf.reduce_sum(targets)
#     ctp = tf.reduce_sum(preds[targets==1])
#     cfp = tf.reduce_sum(preds[targets==0])
#     beta_squared = beta * beta

#     c_precision = ctp / (ctp + cfp )
#     c_recall = ctp / (y_true_count )
#     dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
#     # print(c_precision)
#     # print(c_recall)
#     # print(dice)

#     return dice
fb=keras.metrics.FBetaScore(beta=0.5,threshold=0.5,average='micro')

def fbeta(mask, mask_pred):
    # mask = tf.cast(tf.squeeze(mask,[-1]), tf.float32)
    # mask_pred = tf.squeeze(mask_pred)
    mask = tf.cast(tf.reshape(mask,(1,128*128*2)), tf.float32)
    # print(mask.shape)
    mask_pred = tf.cast(tf.reshape(mask_pred,(1,128*128*2)), tf.float32)
    # print(mask.shape)
    # print(tf.reduce_max(mask))
    # print(tf.reduce_max(mask_pred))
    # print(mask_pred.shape)

    #th =0.5
    #dice = fbeta_numpy(mask, tf.cast((mask_pred >= th), tf.int32), beta=0.5)
    
    fb.update_state(mask, mask_pred)
    dice=fb.result()
    #print(dice.numpy())
    #print(dice)
    return dice


# Compile model.
# initial_learning_rate = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True
# )

optimizer=keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss=['binary_crossentropy'],
              optimizer=optimizer, metrics=['accuracy', recall, precision, fbeta])



# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.hdf5", save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_fbeta", patience=30)


history = model.fit(train_dataset,
    validation_data=validation_dataset, batch_size=10, epochs=100, verbose=2,
    callbacks=[checkpoint_cb,early_stopping_cb])


# # Evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# # Loads the weights
# model.load_weights(checkpoint_path)

# # Re-evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#model.save("model.keras")
#loaded_model = keras.saving.load_model("model.keras")

#%%

history = model.history

#
# plt.plot(history.history['fbeta'],
#          label='Training')
# plt.plot(history.history['val_fbeta'],
#           label='Validation')
# plt.ylabel('f_0.5')
# plt.legend()
# plt.show()

##
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
ax = ax.ravel()

for i, metric in enumerate([ "loss","fbeta",'precision_2','recall_2']):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

#%%

# Load best weights.
model.load_weights("3d_image_classification.hdf5")
count=[]
for i in range(40):
    prediction = model.predict(np.expand_dims(x_val[i], axis=0))[0]
    a,c1=np.unique(prediction[:, :, 0],return_counts=True)
    count.append(c1)
    
prediction = model.predict(np.expand_dims(x_val[39], axis=0))[0]   
plt.imshow(prediction, cmap="gray")

plt.imshow(y_val[39], cmap="gray")

print(y_val[0].shape)
print(prediction.shape)

print(recall(y_val[0],prediction))


data = train_dataset.take(1)
images, labels = list(data)[0]
print(images[0].shape)
print(labels[0].shape)
prediction1 = model.predict(images)[0]
print(prediction1.shape)

images = images.numpy()
image = images[0]
#print("Dimension of the X ray scan slice is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 0]), cmap="gray")
plt.imshow(np.squeeze(prediction1.numpy()[:, :, 0]), cmap="gray")
plt.imshow(np.squeeze(labels[0].numpy()[:, :, 0]), cmap="gray")


# mask = tf.cast(tf.reshape(labels[0],(1,-1)), tf.float32)
# print(mask.shape)
# mask_pred = tf.reshape(prediction1,(1,-1))

# #th =0.5
# #dice = fbeta_numpy(mask, tf.cast((mask_pred >= th), tf.int32), beta=0.5)
# fb=keras.metrics.FBetaScore(beta=0.5,threshold=0.5,average='micro')
# fb.update_state(mask, mask_pred)
# dice=fb.result()
# print(dice.numpy())

