from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests

from generators.generator import DataGenerator
from models.models_text import *
from models.models_image import *

#Instance
app = FastAPI()

#Origins
origins = ["https://localhost:8000",]

#Constantes
BATCH_SIZE_IMAGE = 32
BATCH_SIZE_TEXT = 64
EPOCHS_TEXT= 15
EPOCHS_IMAGE = 5
EPOCHS_FUSION = 15
NUM_FOLDS = 3
NUM_TARGETS = 84916
TARGET_SHAPE = [100, 100, 3]
TEST_SPLIT= .16
RANDOM_STATE = 123
NUM_TARGETS_TRAIN = int(NUM_TARGETS*(1 - TEST_SPLIT))
NUM_TARGETS_TEST = int(NUM_TARGETS*(TEST_SPLIT))
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250
MAX_TOKENS=10000

#Modeles 
model_text = ModelText_Neural_Simple(
        suffix="_stemmed_translated",
        load=True,
    )

model_image = ModelImage_CNN_Lenet(
        suffix="_050",
        load=True,
    )

#DataGenrator
data_generator = DataGenerator(
    from_api=True,
    target_shape=(50,50,3)
)

@app.get("/api/image/layer/{idx_layer}")
def get_layer(idx_layer):
    model_text = ModelText_Neural_Simple(
        suffix="_translated",
        load=True,
    )
    layers = model_text.model.layers

    return StreamingResponse(image.read(), media_type="image/jpeg")


@app.get("/api/image/predict/{url:path}")
def pred_image(image_url: str):
    UPLOADED_PATH = 'uploaded_image.jpg'
    img_data = requests.get(str(image_url)).content
    with open(UPLOADED_PATH, 'wb') as handler:
        handler.write(img_data)

    im = data_generator.load_image(UPLOADED_PATH)
    
    im = im.numpy().reshape(1,50, 50, 3)
    response = model_image.predict(
        im, 
        generator=data_generator,
        for_api=True,
        is_="image"
    )
    print("got response from fastapi")
    return response

@app.get("/api/text/predict/{input}")
def text_prediction(input):
    
    response = model_text.predict(
        input.split(";"), 
        generator=data_generator,
        for_api=True,
        is_="text"
        )
    print("got response from fastapi")
    return response



