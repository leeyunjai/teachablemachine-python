import tensorflow as tf
import numpy as np
import time
import cv2
import argparse
import os
import sys
sys.path.append('/home/pi/pibo-edu/lib')
import vision.stream as stream

def predict(img):
  img = cv2.resize(img, (224,224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  normalized_image_array = (img.astype(np.float32) / 127.0) - 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  data[0] = normalized_image_array
  prediction = model.predict(data)
  return prediction

if __name__ == "__main__":

  # teachable machine 
  # Export Model(tensorflow/Keras - Download my model)
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='filename or video', default=0)
  parser.add_argument('--model', help='model path', default="keras_model.h5")
  parser.add_argument('--labels', help='label path', default="labels.txt")
  args = parser.parse_args()

  os.system("v4l2-ctl -c vertical_flip=1,horizontal_flip=1,white_balance_auto_preset=3")

  with open(args.labels) as f:
    class_names = f.read().split('\n')
    class_names = [class_names[i][class_names[i].index(' ')+1:] for i in range(len(class_names)-1)]

  model = tf.keras.models.load_model(args.model)

  if str(args.input).isdigit() == True:
    cap = cv2.VideoCapture(int(args.input))

    while True:
      img = cap.read()[1]
      p = predict(img)[0]
      idx = np.argmax(predict(img)[0])
      print(p, idx, class_names[idx])

      cv2.putText(img, class_names[idx], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)

      cv2.imshow("Teachable Machine Viewer", img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
  else:
    img = cv2.imread(args.input)
    print(predict(img))
