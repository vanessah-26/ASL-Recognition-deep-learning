import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

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
        img = np.array(img) #img contains pixels of that img
        images.append(img)
        labels.append(label)

data = {'Image': images, 'Label': labels}
df = pd.DataFrame(data)

#
sample_size = 5  #number of images to display
random_indices = random.sample(range(len(df)), sample_size)

#display the selected images
for i in random_indices:
    img_data = df.at[i, 'Image']

    #check and adjust the image data as needed
    img_data = img_data.astype(np.float32) / 255.0  #normalize pixel values

    plt.imshow(img_data, cmap='gray')
    plt.title(f"Image {i}")
    plt.show()

#df Dimension

num_samples, num_features = df.shape

print(f'Number of samples (rows): {num_samples}')
print(f'Number of features (columns): {num_features}')
#columns: Image and Label

#img Dimension
for i, row in df.iterrows():
  img_data = row['Image']
  height, width, _ = img_data.shape #_ to ignore color channel
  print(f"image {i} has height x width of {height} x {width}")

#count img
label_counts = {}  #Dictionary to store counts for each label

for folder in os.listdir(asl_path):
    folder_path = os.path.join(asl_path, folder)
    if os.path.isdir(folder_path): #check if the crrent folder_path is a subdir
        label = folder
        image_count = 0
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if img_path.endswith('.jpeg'):
                image_count += 1
        label_counts[label] = image_count

#Print the counts for each label
for label, count in label_counts.items():
    print(f"Label: {label}, Image Count: {count}")

#normalize images
for i, row in df.iterrows():
  img_data = row['Image']
  img_data = img_data.astype(np.float32) #cast img_data from float64 to float32
  normalized_img = img_data/255.0
  df.at[i, 'Image'] = normalized_img #update each img to rescale

#check if normalization is correct
for i, row in df.iterrows():
  img_data = row['Image']
  min_pixel = np.min(img_data)
  max_pixel = np.max(img_data)
  print(f"Image {i}: Min pixel value: {min_pixel}, Max pixel value: {max_pixel}")

#show img of each label 


uniq_labels = df['Label'].unique()
num_img_per_row = 5

selected_rows = [] #store random rows/1 img of each letter

for label in uniq_labels:
    label_row = df[df['Label'] == label].sample(1).iloc[0] # take 1 random sample img of each label
    selected_rows.append(label_row) #append it to the created list

num_display_rows = len(uniq_labels) // num_img_per_row + 1 #calculate how many rows we need and +1 is to ensure that every row has 5 imgs as i wanted
num_display_cols = num_img_per_row

fig, axs = plt.subplots(num_display_rows, num_display_cols, figsize=(20, 10))

#get images and labels to display
for i, row in enumerate(selected_rows):
    img_data = row['Image']
    label = row['Label']

    img_data = img_data.astype(np.float32) #/ 255.0
    # /255.0 is for normalizing all images to scale 0-1 for model training
    # but it will only show black background when visualize (because defined cmap='gray' and grayscale images )

    axs[i // num_img_per_row, i % num_img_per_row].imshow(img_data, cmap='gray') #decide which row the subplot should be in, and which column
    axs[i // num_img_per_row, i % num_img_per_row].set_title(f'Label: {label}')

#delete extra subplots (additional subplots when all img of each label already displayed)
for i in range(len(uniq_labels), num_display_rows * num_img_per_row):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()


#B&W img/class
uniq_labels = df['Label'].unique()
num_img_per_row = 5

selected_rows = [] #store random rows/1 img of each letter

for label in uniq_labels:
    label_row = df[df['Label'] == label].sample(1).iloc[0] # take 1 random sample img of each label
    selected_rows.append(label_row) #append it to the created list

num_display_rows = len(uniq_labels) // num_img_per_row + 1 #calculate how many rows we need and +1 is to ensure that every row has 5 imgs as I wanted
num_display_cols = num_img_per_row

fig, axs = plt.subplots(num_display_rows, num_display_cols, figsize=(20, 10))

#get images and labels to display
for i, row in enumerate(selected_rows):
    img_data = row['Image']
    label = row['Label']

    img_data = img_data.astype(np.float32)
    img_data = np.mean(img_data, axis=2)
    #convert to grayscale by calculating the mean at third axis (axis=2)--> color channel
    # => calculate the avg value of color channels at each pixel
    # after this we'll have the greyscale ver of each img which will be shown using imshow, cmap='gray'

    axs[i // num_img_per_row, i % num_img_per_row].imshow(img_data, cmap='gray') #decide which row the subplot should be in, and which column and show img
    axs[i // num_img_per_row, i % num_img_per_row].set_title(f'Label: {label}')

#delete extra subplots (additional subplots when all img of each label already displayed)
for i in range(len(uniq_labels), num_display_rows * num_img_per_row):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()

#label encode
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

alphabetical_labels = df['Label']
numerical_labels = le.fit_transform(alphabetical_labels)

df['Numerical Label'] = numerical_labels #update the numerical labels into df

uniq_labels = df['Label'].unique()

for label in uniq_labels:
  num_label = df[df['Label'] == label]['Numerical Label'].iloc[0] #select numerical labels by mapping between orignial label and its corresponding numerical label
  print(f"Original label: {label}, Numerical Label: {num_label}")


#info
from PIL import Image

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2515 entries, 0 to 2514
# Data columns (total 3 columns):
#  #   Column           Non-Null Count  Dtype 
# ---  ------           --------------  ----- 
#  0   Image            2515 non-null   object
#  1   Label            2515 non-null   object
#  2   Numerical Label  2515 non-null   int64 
# dtypes: int64(1), object(2)
# memory usage: 59.1+ KB


resized_images = []

for img in df['Image']:
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  img = img.resize((224,224))
  img = np.array(img)
  resized_images.append(img)

df['Image'] = resized_images
for img in df['Image']:
  print(img.shape)

#should print:
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# ...

