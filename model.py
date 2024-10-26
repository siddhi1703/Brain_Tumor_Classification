import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        data_dir, 
        target_size=(224, 224), 
        class_mode='binary', 
        subset='training'
    )
    val_data = datagen.flow_from_directory(
        data_dir, 
        target_size=(224, 224), 
        class_mode='binary', 
        subset='validation'
    )
    return train_data, val_data

def train_model(model, train_data, val_data):
    history = model.fit(train_data, epochs=50, validation_data=val_data)
    return history

if __name__ == "__main__":
    data_dir = 'brain_mri_dataset'  # Update to your dataset path
    model = create_model()
    train_data, val_data = load_data(data_dir)
    train_model(model, train_data, val_data)
    model.save('brain_tumor_model.h5')
