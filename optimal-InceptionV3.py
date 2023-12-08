import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout 
from sklearn.preprocessing import LabelEncoder

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

#split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)
#
X_train = X_train/255.0
X_test = X_test/255.0


le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

#import base model
inception_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#fine tune
model_inceptionv3 = Sequential([
    inception_model,
    GlobalAveragePooling2D(),
    Dense(245, activation = 'relu', kernel_regularizer=l2(0.01)),
    Dropout(0.9),
    Dense(len(le.classes_), activation = 'softmax')
])

model_inceptionv3.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#training and testing
train_inception = model_inceptionv3.fit(X_train, y_train_encoded, epochs = 6, batch_size = 36,validation_data = (X_test, y_test_encoded))

inception_loss, test_inception = model_inceptionv3.evaluate(X_test, y_test_encoded)
print(f"InceptionV3 Accuracy: {test_inception}")

#visualize result
plt.plot(train_inception.history['accuracy'], label = 'Training Accuracy')
plt.plot(train_inception.history['val_accuracy'], label = 'Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.2, 1.0)

plt.legend()
plt.show()

plt.plot(train_inception.history['loss'], label = 'Training Loss')
plt.plot(train_inception.history['val_loss'], label = 'Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.2, 1.0)

plt.legend()
plt.show()

# show confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns


y_pred = model_inceptionv3.predict(X_test)

#convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

confusion = confusion_matrix(y_test_encoded, y_pred_classes)


plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#other metrics 
from sklearn.metrics import classification_report

classification_rep = classification_report(y_test_encoded, y_pred_classes, target_names=le.classes_)
print(classification_rep)

#               precision    recall  f1-score   support

#            0       0.53      1.00      0.69        21
#            1       0.86      1.00      0.92        18
#            2       1.00      1.00      1.00        19
#            3       1.00      1.00      1.00        24
#            4       0.95      1.00      0.98        21
#            5       1.00      0.96      0.98        25
#            6       1.00      0.96      0.98        27
#            7       1.00      1.00      1.00        20
#            8       1.00      0.94      0.97        17
#            9       1.00      1.00      1.00        15
#            a       1.00      1.00      1.00        16
#            b       1.00      1.00      1.00        28
#            c       1.00      1.00      1.00        25
#            d       1.00      1.00      1.00        15
#            e       1.00      1.00      1.00        17
#            f       1.00      1.00      1.00        23
#            g       1.00      1.00      1.00        22
#            h       1.00      1.00      1.00        21
#            i       1.00      1.00      1.00        18
#            j       1.00      1.00      1.00        12
#            k       0.95      1.00      0.98        20
#            l       1.00      1.00      1.00        25
#            m       1.00      1.00      1.00        25
#            n       1.00      1.00      1.00        27
#            o       0.00      0.00      0.00        19
#            p       1.00      1.00      1.00        24
#            q       1.00      1.00      1.00        20
#            r       1.00      1.00      1.00        26
#            s       1.00      1.00      1.00        25
#            t       1.00      1.00      1.00        14
#            u       1.00      1.00      1.00        19
#            v       1.00      1.00      1.00        16
#            w       0.96      1.00      0.98        22
#            x       1.00      1.00      1.00        23
#            y       1.00      1.00      1.00        27
#            z       1.00      0.84      0.91        19

#     accuracy                           0.97       755
#    macro avg       0.95      0.96      0.96       755
# weighted avg       0.95      0.97      0.96       755

#show example images of prediction vs true 
import random

#select a random sample of test imgs and corresponding labels
sample_size = 5
random_indices = random.sample(range(len(X_test)), sample_size)
sample_images = X_test[random_indices]
sample_labels = y_test_encoded[random_indices]

#make predictions for the sample imgs
sample_predictions = model_inceptionv3.predict(sample_images)
sample_pred_classes = np.argmax(sample_predictions, axis=1)

#display the sample imgs with true and predicted labels
plt.figure(figsize=(15, 5))
for i in range(sample_size):
    plt.subplot(1, sample_size, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {le.classes_[sample_labels[i]]}\nPredicted: {le.classes_[sample_pred_classes[i]]}")
    plt.axis('off')
plt.show()


#show a class activation map (show what features model focused on to predict)

#overlay heatmap on the OG img
final_image = cv2.addWeighted(
    sample_image.astype(np.uint8),
    0.5,
    heatmap_on_image_uint8,
    0.5,
    0
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_image)
plt.title(f'True Class: {le.classes_[y_test_encoded[sample_index]]}\nPredicted Class: {le.classes_[predicted_class]}')

plt.subplot(1, 2, 2)
plt.imshow(final_image)
plt.title('Class Activation Map')
plt.colorbar()
plt.show()


