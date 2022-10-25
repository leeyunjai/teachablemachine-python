import cv2
import numpy as np
from PIL import Image

class TeachableMachineKeras:
  def __init__(self):
    pass

  def load(self, model_path, label_path):
    from tensorflow import keras

    with open(label_path) as f:
      c = f.read().split('\n')
      class_names = [c[i].split(maxsplit=1)[1] for i in range(len(c)-1)]

    self.model = keras.models.load_model(model_path, compile=False)
    self.class_names = class_names

  def predict(self, img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1
    predictions = self.model.predict(img)[0]
    return predictions, self.class_names[np.argmax(predictions)]

class TeachableMachineTf:
  def __init__(self):
    pass

  def load(self, model_path, label_path):
    from tflite_runtime.interpreter import Interpreter

    with open(label_path, 'r') as f:
      c = f.readlines()
      class_names = [item.split(maxsplit=1)[1].strip('\n') for item in c]
    self.class_names = class_names

    # Load TFLite model and allocate tensors
    self.interpreter = Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()

    # Get input and output tensors.
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    # check the type of the input tensor
    self.floating_model = self.input_details[0]['dtype'] == np.float32

    self.height = self.input_details[0]['shape'][1]
    self.width = self.input_details[0]['shape'][2]

  def predict(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (self.width, self.height))
    image = Image.fromarray(img)

    # Add a batch dimension
    input_data = np.expand_dims(image, axis=0)

    if self.floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5

    # feed data to input tensor and run the interpreter
    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    self.interpreter.invoke()

    # Obtain results and map them to the classes
    preds = self.interpreter.get_tensor(self.output_details[0]['index'])
    preds = np.squeeze(preds)
    return preds, self.class_names[np.argmax(preds)]

if __name__ == "__main__":

    tk = TeachableMachineTf()
    tk.load("./tflite_model/model_unquant.tflite", "./tflite_model/labels.txt")
    
    #tk = TeachableMachineKeras()
    #tk.load("./keras_model/keras_model.h5", "./keras_model/labels.txt")

    cap = cv2.VideoCapture(0)
    
    while True:
        s = time.time()
        _, img = cap.read()
        result = tk.predict(img)
        print(time.time()-s, result)
