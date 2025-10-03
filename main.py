# Name: Aya Ahmed Mohamed Yousef Borham 
# Section: 2 
# Department: CS
# Grade: Four
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model("keras_Model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

image = Image.open("test.jpg").convert("RGB")  
size = (224, 224)  
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.expand_dims(normalized_image_array, axis=0)

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print("Class:", class_name.strip())
print("Confidence Score:", confidence_score)
