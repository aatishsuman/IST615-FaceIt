# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:27:34 2020

@author: Aatish
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image

class MaskModel:
    
    def __init__(self):
        self.WITH_MASK_DIR = 'with_mask'
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return 1 if parts[-2] == self.WITH_MASK_DIR else 0

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(tf.cast(img, tf.float32) / 255., [self.IMG_HEIGHT, self.IMG_WIDTH])
    
    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return tf.data.Dataset.from_tensors((img, label))
    
    def generate_dataset(self, images_dir, batch_size, validation_ratio):
        list_ds = tf.data.Dataset.list_files(os.path.join(images_dir, 'train') + os.path.sep + '*' + os.path.sep + '*', shuffle=False)
        image_count = len(list(list_ds))
        list_ds = list_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=False)
        dataset = list_ds.interleave(self.process_path, num_parallel_calls=self.AUTOTUNE)
        validation_size = int(image_count * validation_ratio)
        train_dataset = dataset.skip(validation_size).cache().batch(batch_size).prefetch(self.AUTOTUNE)
        validation_dataset = dataset.take(validation_size).cache().batch(batch_size).prefetch(self.AUTOTUNE)
        return train_dataset, validation_dataset
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                                            tf.keras.layers.MaxPooling2D(2, 2),
                                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                            tf.keras.layers.MaxPooling2D(2, 2),
                                            tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation='relu'),
                                            tf.keras.layers.Dense(1, activation='sigmoid')])
        self.model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())
    
    def train(self, images_dir, batch_size, validation_ratio, num_epochs):
        train_dataset, validation_dataset = self.generate_dataset(images_dir, batch_size, validation_ratio)
        self.create_model()
        return self.model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)
    
    def save_model(self, folder, name):
        model_json = self.model.to_json()
        with open(os.path.join(folder, name + '.json'), 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(folder, name + '.h5'))
        print('Saved model to disk')
    
    def load_model(self, folder, name):
        json_file = open(os.path.join(folder, name + '.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(os.path.join(folder, name + '.h5'))
        print('Loaded model from disk')
    
    def predict_from_folder(self, test_dir):
        file_names = os.listdir(test_dir)
        results = []
        for file_name in file_names:
            img = image.load_img(os.path.join(test_dir, file_name), target_size=(self.IMG_HEIGHT, self.IMG_WIDTH))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = np.vstack([img])
            results.append((file_name, self.model.predict(img)[0]))
        return results
    
    def predict_from_base64(self, img_str):
        img = image.img_to_array(Image.open(BytesIO(base64.b64decode(img_str))).resize((self.IMG_HEIGHT, self.IMG_WIDTH)))
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        return self.model.predict(img)[0]
    
def main():
    IMAGES_DIR = os.path.join('data', 'images')
    BATCH_SIZE = 32
    VALIDATION_RATIO = 0.2
    NUM_EPOCHS = 5
    model = MaskModel()
    model.train(IMAGES_DIR, BATCH_SIZE, VALIDATION_RATIO, NUM_EPOCHS)
    model.save_model('model', 'model')

if __name__ == '__main__':
    main()
    
    