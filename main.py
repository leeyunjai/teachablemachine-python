from tmlib import *
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--type', help='tflite or keras', choices=['tflite', 'keras'], required=True)
  parser.add_argument('--model', help='keras_model.h5 or model_unquant.tflite', required=True)
  parser.add_argument('--labels', help='labels.txt', required=True)
  args = parser.parse_args()

  tm = TeachableMachineTf() if args.type == 'tflite' else TeachableMachineKeras()
  tm.load(args.model, args.labels)

  cap = cv2.VideoCapture(0)
  while True:
    _, img = cap.read()
    res, name = tm.predict(img)

    print("{}: {:.2f}%".format(name,np.max(res)*100))
    cv2.putText(img, "{}: {:.2f}%".format(name,np.max(res)*100), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 4)
    cv2.imshow("Teachable Machine Viewer", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cap.release()
      break
