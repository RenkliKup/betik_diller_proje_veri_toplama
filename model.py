import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

train_dir = "./output/train/"
val_dir="./output/val/"
"""
    # Use `ImageDataGenerator` to rescale the images.
    # Create the train generator and specify where the train dataset directory, image size, batch size.
    # Create the validation generator with similar approach as the train generator with the flow_from_directory() method.
"""

# We are trying to minimize the resolution of the images without loosing the 'Features'
# For facial recognition, this seems to be working fine, you can increase or decrease it
IMAGE_SIZE = 224

# Depending upon the total number of images you have set the batch size
# I have 50 images per person (which still won't give very accurate result)
# Hence, I am setting my batch size to 5
# So in 10 epochs/iterations batch size of 5 will be processed and trained
BATCH_SIZE = 5

# We need a data generator which rescales the images
# Pre-processes the images like re-scaling and other required operations for the next steps
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2)

# We separate out data set into Training, Validation & Testing. Mostly you will see Training and Validation.
# We create generators for that, here we have train and validation generator.
# Create a train_generator
train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

# Create a validation generator
val_generator = data_generator.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

# Triggering a training generator for all the batches

print(train_generator.class_indices)
# This will print all classification labels in the console



IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2 Hidden and output layer is in the model The first
# layer does a redundant work of classification of features which is not required to be trained
# (This is also called as bottle neck layer)

# Hence, creating a model with EXCLUDING the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,  # 1
    tf.keras.layers.Conv2D(32, 2, activation='relu'),  # 2
    tf.keras.layers.Dropout(0.2),  # 3
    tf.keras.layers.GlobalAveragePooling2D(),  # 4
    tf.keras.layers.Dense(2, activation='softmax')  # 5
])

model.compile(optimizer=tf.keras.optimizers.Adam(),  # 1
              loss='categorical_crossentropy',  # 2
              metrics=['accuracy'])  # 3

model.summary()

# Printing some statistics
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

# Train the model
# We will do it in 10 Iterations
epochs = 5

# Fitting / Training the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator)



model.save("first_model")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()