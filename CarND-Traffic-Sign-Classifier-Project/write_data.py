import numpy as np
import cv2
import glob
import pickle

X_images = []
y_labels = []
for myfile in glob.glob("jpgs/t*.jpg"):
  image = cv2.imread(myfile)
  image = cv2.resize(image, (32, 32))
  X_images.append(image)
  y_labels.append(myfile[6:8])
X_images = np.array(X_images)
y_labels = np.array(y_labels)
X_data = {"features": X_images, "labels": y_labels}
with open('dataset/predict.p','wb') as f:
  pickle.dump(X_data,f)
