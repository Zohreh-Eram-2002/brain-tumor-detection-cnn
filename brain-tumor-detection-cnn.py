# # **1- Overview :**
# ### **in this project we will implement the below paper**
# 
# https://www.researchgate.net/publication/362504981_Brain_Tumor_Detection_using_MRI_Images_and_Convolutional_Neural_Network
# 
# > **Abstract** : A brain tumor is the cause of abnormal growth of
# cells in the brain. Magnetic resonance imaging (MRI) is the most
# practical method for detecting brain tumors. Through these
# MRIs, doctors analyze and identify abnormal tissue growth and
# can confirm whether the brain is affected by a tumor or not.
# Today, with the emergence of artificial intelligence techniques,
# the detection of brain tumors is done by applying the techniques
# and algorithms of machine learning and deep learning. The
# advantages of the application of these algorithms are the quick
# prediction of brain tumors, fewer errors, and greater precision,
# which help in decision-making and in choosing the most
# appropriate treatment for patients. In the proposed work, a
# convolution neural network (CNN) is applied with the aim of
# detecting the presence of a brain tumor and its performance is
# analyzed. The main purpose of this article is to adopt the
# approach of convolutional neural networks as a machine learning
# technique to perform brain tumor detection and classification.
# Based on training and testing results, the pre-trained
# architecture model reaches 96% in precision and classification
# accuracy rates. For the given dataset, CNN proves to be the
# better technique for predicting the presence of brain tumors.

# # **2- Setup**

# ### Import Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#deep learning and computer vision
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ### Constants And Parameters

DATASET_PATH = "/content/brain_tumor_dataset/Brain_Tumor_Detection"

#Data Parameter
CLASS_LABELS = {"yes": 1, "no": 0}
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

#CNN Network HyperParameters

# # **3- Load Data**

# ### Download Data From Kaggle And Unzip ( https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri/code)

!curl -L -o  brain-tumor-detection-mri.zip https://www.kaggle.com/api/v1/datasets/download/abhranta/brain-tumor-detection-mri

!unzip /content/brain-tumor-detection-mri.zip -d brain_tumor_dataset

# ### Load Images (Yes / No From DATASET Path)

data = []
labels = []

for label, value in CLASS_LABELS.items():
    data_folder = os.path.join(DATASET_PATH, label)

    for filename in os.listdir(data_folder):

        img_path = os.path.join(data_folder, filename)

        #read image
        img = cv2.imread(img_path)
        #resize
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        #normalize image
        img = img / 255.0

        data.append(img)
        labels.append(value)


data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print("Number of MRI Images : ",len(data))

# ### Show sample of images

sample_indicies = np.random.randint(0,3000,9)


fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(14,10))
counter= 0
for row in range(3) :
   for col in range(3):
      axes[row][col].grid(False)
      axes[row][col].axis("off")
      axes[row][col].imshow(data[sample_indicies[counter]])
      axes[row][col].set_title(labels[sample_indicies[counter]])
      counter +=1


# # **4- Prepare Data For Traning Model**

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#dataset shapes
print(f"training data : {X_train.shape}, training labels : {y_train.shape}")
print(f"validation data : {X_val.shape}, validation labels : {y_val.shape}")
print(f"testing data : {X_test.shape}, testing labels : {y_test.shape}")

# # **5- Build CNN Model (with Keras)**

# ### Build Model

model = Sequential()

# conv Layers
model.add(Conv2D(20, (4, 4), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(20, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(20, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

#flat and dense layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  #output layer ===> two class [0,1] == [yes, no]


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


keras.utils.plot_model(model,show_shapes=True)

# ### Train Model

#train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val,y_val))


# # **6- Model Evaluation**

#plot traning acc
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

#plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()




test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f"test Accuracy: {test_accuracy * 100:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test

pd.DataFrame(classification_report(y_true, y_pred_classes,output_dict=True)).T

# ### Confusion Matrix

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


sample_indicies = np.random.randint(0, len(X_test),9  )

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, idx in enumerate(sample_indicies):
    axes[i].imshow(X_test[idx])
    axes[i].set_title(f'True: {y_test[idx]}, Model Prediction: {y_pred_classes[idx]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

