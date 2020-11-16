# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:27:34 2020

@author: Aatish
"""

import numpy as np
import os
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import base64

class MaskModel:
    
    def __init__(self):
        self.WITH_MASK_DIR = 'with_mask'
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.faces = {'with_mask': 0, 'without_mask': 0}
    
    def extract_face(self, filename, faces_dir):
        label = filename.split(os.path.sep)[-2]
        pixels = cv2.imread(filename)
        classifier = cv2.CascadeClassifier(os.path.join('model', 'haarcascade_frontalface_default.xml'))
        bboxes = classifier.detectMultiScale(pixels)
        try:
            x, y, width, height = bboxes[0]
            face = pixels[y: y + height, x: x + width]
            name = os.path.join(faces_dir, label, filename.split(os.path.sep)[-1])
            cv2.imwrite(name, cv2.resize(face, (self.IMG_HEIGHT, self.IMG_WIDTH)))
            cv2.destroyAllWindows()
            self.faces[label] += 1
        except:
            cv2.destroyAllWindows()
    
    def generate_faces(self, images_dir, faces_dir):
        print('Generating faces for training')
        try:  
            os.mkdir(faces_dir)
            os.mkdir(os.path.join(faces_dir, 'with_mask'))
            os.mkdir(os.path.join(faces_dir, 'without_mask'))
        except OSError as error:  
            print(error)
        images_path = os.path.join(images_dir, 'train') + os.path.sep + '*' + os.path.sep + '*.jpg'
        [self.extract_face(path, faces_dir) for path in glob.glob(images_path)]
        print('Total faces generated: ', self.faces)
    
    def generate_dataset(self, images_dir, faces_dir, batch_size, validation_ratio):
        self.generate_faces(images_dir, faces_dir)
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
          os.path.join(faces_dir),
          validation_split=validation_ratio,
          subset='training',
          seed=42,
          image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
          batch_size=batch_size)        
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
          os.path.join(faces_dir),
          validation_split=validation_ratio,
          subset='validation',
          seed=42,
          image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
          batch_size=batch_size)
        train_dataset = train_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        return train_dataset, validation_dataset
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                                            tf.keras.layers.MaxPooling2D(2, 2),
                                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.MaxPooling2D(2, 2),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation='relu'),
                                            tf.keras.layers.Dense(1, activation='sigmoid')])
        self.model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())
    
    def train(self, images_dir, faces_dir, batch_size, validation_ratio, num_epochs):
        train_dataset, validation_dataset = self.generate_dataset(images_dir, faces_dir, batch_size, validation_ratio)
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
            pixels = cv2.resize(cv2.imread(os.path.join(test_dir, file_name)), (self.IMG_HEIGHT, self.IMG_WIDTH))
            classifier = cv2.CascadeClassifier(os.path.join('model', 'haarcascade_frontalface_default.xml'))
            bboxes = classifier.detectMultiScale(pixels)
            try:
                x, y, width, height = bboxes[0]
                face = pixels[y: y + height, x: x + width]
            except:
                face = pixels
            img = cv2.resize(face, (self.IMG_HEIGHT, self.IMG_WIDTH))
            img = np.expand_dims(np.asarray(img), axis=0)
            img = np.vstack([img])
            results.append(int(self.model.predict(img)[0] > 0.5))
        return results
    
    def predict_from_base64(self, img_str):
        pixels = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(img_str), dtype=np.uint8), 1), (self.IMG_HEIGHT, self.IMG_WIDTH))
        classifier = cv2.CascadeClassifier(os.path.join('model', 'haarcascade_frontalface_default.xml'))
        bboxes = classifier.detectMultiScale(pixels)
        try:
            x, y, width, height = bboxes[0]
            face = pixels[y: y + height, x: x + width]
        except:
            face = pixels
        img = cv2.resize(face, (self.IMG_HEIGHT, self.IMG_WIDTH))
        img = np.expand_dims(np.asarray(img), axis=0)
        img = np.vstack([img])
        return int(self.model.predict(img)[0] > 0.5)
    
def main():
    IMAGES_DIR = os.path.join('data', 'images')
    FACES_DIR = os.path.join(IMAGES_DIR, 'train', 'faces')
    BATCH_SIZE = 32
    VALIDATION_RATIO = 0.2
    NUM_EPOCHS = 2
    model = MaskModel()
    model.train(IMAGES_DIR, FACES_DIR, BATCH_SIZE, VALIDATION_RATIO, NUM_EPOCHS)
    model.save_model('model', 'model')

if __name__ == '__main__':
    main()
    
    