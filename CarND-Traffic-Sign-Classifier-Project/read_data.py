import numpy as np
import cv2
import glob
import pickle

with open('predict.p', 'rb') as f:
   X_data = pickle.load(f)
   X_predict = X_data["features"]
   y_labels = X_data["labels"]
   n = len(X_predict)
   cv2.namedWindow('image', cv2.WINDOW_NORMAL)
   for i in range(n):
      print("label is {}".format(y_labels[i]))
      cv2.imshow('image', X_predict[i])
      cv2.waitKey(0)
   cv2.destroyAllWindows()
