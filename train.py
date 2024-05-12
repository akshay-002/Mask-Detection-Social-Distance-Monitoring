import numpy as np
import cv2
from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Load the cascade classifier
face_model = cv2.CascadeClassifier('/Users/akshay/projects/face_mask_model/haarcascade_frontalface_default.xml')

# Define directories for training, testing, and validation datasets
train_dir = '/Users/akshay/projects/face_mask_model/archive/Face_Mask_Dataset/Train'
test_dir = '/Users/akshay/projects/face_mask_model/archive/Face_Mask_Dataset/Test'
val_dir = '/Users/akshay/projects/face_mask_model/archive/Face_Mask_Dataset/Validation'

# Data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                    class_mode='categorical', batch_size=32)
# Data generators for validation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128), class_mode='categorical',
                                                  batch_size=32)
# Data generators for testing
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128), class_mode='categorical',
                                                   batch_size=32)

# Load the VGG19 model with weights pre-trained on ImageNet dataset
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Set all layers of the VGG19 model to non-trainable
for layer in vgg19.layers:
    layer.trainable = False

model = Sequential()  # Create a Sequential model
model.add(vgg19)  # Add the VGG19 base model
model.add(Flatten())  # Flatten the output of the VGG19 base model

# Add a dense layer with softmax activation function for binary classification.
# Although sigmoid activation is typically recommended for binary classification tasks,
# softmax activation is used here to ensure that the probabilities sum up to one.
model.add(Dense(2, activation='softmax'))
model.summary()

# Compile the model with Adam optimizer, categorical cross entropy loss, and accuracy metric
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

# Train the model for 10 epochs using the training and validation generators.
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator) // 32,
                    epochs=10, validation_data=val_generator,
                    validation_steps=len(val_generator) // 32)

# Get the final training accuracy and loss from the training history.
training_accuracy = history.history['accuracy'][-1]
training_loss = history.history['loss'][-1]

# Print the training accuracy and loss.
print(f'Training Accuracy of the Netmask Model : {training_accuracy * 100:.2f} %')
print(f'Training Loss of the Netmask Model : {training_loss :.2f} %')

# Evaluate the model on the test data.
test = model.evaluate(test_generator)

# Get the testing accuracy and loss.
testing_accuracy = test[1]
testing_loss = test[0]

# Print the testing accuracy and loss.
print(f'Testing Accuracy of the Netmask Model : {testing_accuracy * 100:.2f} %')
print(f'Testing Loss of the Netmask Model : {testing_loss :.2f} %')

# Save the model to the specified file.
model.save('Netmask_10_epoch.h5')
print("Model saved successfully.")

# %%
