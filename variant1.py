import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import cv2
import random
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten,Dense

#../input/covid-face-mask-detection-dataset/New Masks Dataset/Test

train_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'
test_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test'
val_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'

#with mask
plt.figure(figsize=(12, 7))
for i in range(5):
    sample = random.choice(os.listdir(train_dir + "WithMask/"))
    plt.subplot(1, 5, i + 1)
    img = load_img(train_dir + "WithMask/" + sample)
    plt.subplots_adjust(hspace=0.001)
    plt.xlabel("With Mask")
    plt.imshow(img)
plt.show()
#without mask
plt.figure(figsize=(12, 7))
for i in range(5):
    sample = random.choice(os.listdir(train_dir + "WithoutMask/"))
    plt.subplot(1, 5, i + 1)
    img = load_img(train_dir + "WithoutMask/" + sample)
    plt.subplots_adjust(hspace=0.001)
    plt.xlabel("Without Mask")
    plt.imshow(img)
plt.show()

img_height, img_lenght\
    = 150, 150
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_height, img_lenght),
                                          class_mode="categorical", batch_size=32, subset="training")

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

valid = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_height, img_lenght),
                                          class_mode="categorical", batch_size=32, subset="validation")

mobilenet = MobileNetV2(weights = "imagenet",include_top = False,input_shape=(150,150,3))
for layer in mobilenet.layers:
    layer.trainable = False

model = Sequential()
model.add(mobilenet)
model.add(Flatten())
model.add(Dense(2,activation="sigmoid"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
checkpoint = ModelCheckpoint("moblenet_facemask.h5",monitor="val_accuracy",save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit_generator(generator=train,steps_per_epoch=len(train)// 32,validation_data=valid,
                             validation_steps = len(valid)//32,callbacks=[checkpoint,earlystop],epochs=15)
model.evaluate_generator(valid)
model.save("face_mask.h5")
pred = model.predict_classes(valid)
pred[:15]
#check

#without mask
mask = "../input/with-and-without-mask/"
plt.figure(figsize=(8, 7))
label = {0: "With Mask", 1: "Without Mask"}
color_label = {0: (0, 255, 0), 1: (0, 0, 255)}
cascade = cv2.CascadeClassifier("../input/frontalface/haarcascade_frontalface_default.xml")
count = 0
i = "../input/with-and-without-mask/mask9.jpg"

frame = cv2.imread(i)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in faces:
    face_image = frame[y:y + h, x:x + w]
    resize_img = cv2.resize(face_image, (150, 150))
    normalized = resize_img / 255.0
    reshape = np.reshape(normalized, (1, 150, 150, 3))
    reshape = np.vstack([reshape])
    result = model.predict_classes(reshape)

    if result == 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_label[0], 3)
        cv2.rectangle(frame, (x, y - 50), (x + w, y), color_label[0], -1)
        cv2.putText(frame, label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
    elif result == 1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_label[1], 3)
        cv2.rectangle(frame, (x, y - 50), (x + w, y), color_label[1], -1)
        cv2.putText(frame, label[1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
    # plt.imshow(frame)
plt.show()
cv2.destroyAllWindows()

mask = "../input/with-and-without-mask/"
plt.figure(figsize=(8, 7))
label = {0: "With Mask", 1: "Without Mask"}
color_label = {0: (0, 255, 0), 1: (0, 0, 255)}
cascade = cv2.CascadeClassifier("../input/frontalface/haarcascade_frontalface_default.xml")
count = 0
i = "../input/with-and-without-mask/mask1.jpg"

frame = cv2.imread(i)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in faces:
    face_image = frame[y:y + h, x:x + w]
    resize_img = cv2.resize(face_image, (150, 150))
    normalized = resize_img / 255.0
    reshape = np.reshape(normalized, (1, 150, 150, 3))
    reshape = np.vstack([reshape])
    result = model.predict_classes(reshape)

    if result == 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_label[0], 3)
        cv2.rectangle(frame, (x, y - 50), (x + w, y), color_label[0], -1)
        cv2.putText(frame, label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
    elif result == 1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_label[0], 3)
        cv2.rectangle(frame, (x, y - 50), (x + w, y), color_label[0], -1)
        cv2.putText(frame, label[1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
    # plt.imshow(frame)
plt.show()
cv2.destroyAllWindows()