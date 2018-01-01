from keras.models import load_model
import tensorflow as tf

file = "/Users/antonp/IdeaProjects/testing/minsum/src/main/java/dga/dga_model.h5"
model = load_model(file)
with open('model.json', "w") as json_file:
    json_file.write(model.to_json())

model.save_weights('model_weights.h5')