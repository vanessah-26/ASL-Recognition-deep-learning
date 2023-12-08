import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


asl_path = "/content/drive/MyDrive/Colab Notebooks/asl_dataset/asl_dataset"

images = []
labels = []

for folder in os.listdir(asl_path): #list out all items (files and dir)
  folder_path = os.path.join(asl_path, folder) #get the folder path by joining asl_path and folder (subdir in asl_path)
  #print(f"Processing folder: {label}")
  if os.path.isdir(folder_path):
    label = folder #label = what the folder is named (each folder is named letter or num)
    for filename in os.listdir(folder_path): #list out all items in folder_path dir
      img_path = os.path.join(folder_path, filename)
      #print(f'Processeing image: {img_path}')
      if img_path.endswith('.jpeg'): #check if it is img/filter out files that are not img
        img = Image.open(img_path)
        img = np.array(img.resize((224, 224))) #resize images to match some cnn expected input and also for computational resource
        img = np.array(img)
        images.append(img)
        labels.append(label)

data = {'Image': images, 'Label': labels}
df = pd.DataFrame(data)

X = np.array(images)
y = np.array(labels)

#split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print("X_train before normalize", X_train.shape)
print("y_train before normalize", y_train.shape)

print("X_test before normalize", X_test.shape)
print("y_test before normalize", y_test.shape)

# X_train before normalize (1760, 224, 224, 3)
# y_train before normalize (1760,)
# X_test before normalize (755, 224, 224, 3)
# y_test before normalize (755,)

#normalize
X_train = X_train/255.0
X_test = X_test/255.0

#label encode
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("X_train after normalize", X_train.shape)
print("y_train after normalize", y_train_encoded.shape)
#should print out the same numbers as normalization only rescale the pixel of each img to 0-1, not change the number or feature of images

# X_train after normalize (1760, 224, 224, 3)
# y_train after normalize (1760,)

#CNN architecture 
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape =(224,224,3), padding = 'same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.7), #increased dropout rate to mitigate overfitting
    tf.keras.layers.Dense(36, activation = 'softmax'),

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

#shown on paper (page3)

#training and testing model 
training = model.fit(X_train, y_train_encoded, epochs = 10, batch_size = 32, validation_data = (X_test, y_test_encoded))  #increased epochs for more training

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_acc}")

#data visualization 
import matplotlib.pyplot as plt

plt.plot(training.history['acc'], label = 'Training Accuracy')
plt.plot(training.history['val_acc'], label = 'Testing Accuracy')
plt.title('Custom CNN accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.2, 1.0) #to better show the models performance
# plt.legend(['Train', 'Test'], loc = 'upper left')
plt.legend()
plt.show()

plt.plot(training.history['loss'], label = 'Training Loss')
plt.plot(training.history['val_loss'], label = 'Testing Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.legend(['Train', 'Test'], loc = 'upper left')

plt.show()


