from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests

from generators.generator import DataGenerator
from models.models_text import *
from models.models_image import *
from models.models_fusion import *

UPLOADED_PATH = 'uploaded_image.jpg'
BATCH_SIZE_IMAGE = 32
BATCH_SIZE_TEXT = 64
BATCH_SIZE_FUSION = 128
EPOCHS_TEXT = 50
EPOCHS_IMAGE = 50
EPOCHS_FUSION = 50
NUM_FOLDS = 3
NUM_TARGETS = 84916
TARGET_SHAPE = [224, 224, 3]
TEST_SPLIT = .16
VALID_SPLIT = .16
RANDOM_STATE = 123
VOCAB_SIZE = 50000
SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 200

#Instance
app = FastAPI()

#Origins
origins = ["https://localhost:8000",]


#Modeles 
model_text_obj = ModelText_Neural_Simple(
        suffix=f"",
        epochs=EPOCHS_TEXT,
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        load=True,
        load_embedding=True
    ).start()

#Objet Model Image
model_image_obj = ModelImage_MobileNet(
        suffix=f"_224_crop_255",
        epochs=EPOCHS_IMAGE,
        target_shape=TARGET_SHAPE,
        load=True,
    ).start()

#DataGenrator
data_generator = DataGenerator(
    from_api=True,
    target_shape=(224,224,3),
    crop=True,
    vocab_size=VOCAB_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    layers_folder_path = model_text_obj.layers_folder_path
)


model_fusion = ModelFusion(
        suffix="_mobilenet_224_crop",
        load=True,
).start()



def save_image(url):
    img_data = requests.get(str(url)).content
    with open(UPLOADED_PATH, 'wb') as handler:
        handler.write(img_data)

@app.get("/api/image/predict/url={image_input:path}")
def pred_image(image_input: str):
    
    save_image(image_input)

    response = model_image_obj.predict(
        [UPLOADED_PATH,], 
        generator=data_generator,
        for_api=True,
        is_="image"
    )
    return response

@app.get("/api/text/predict/text={text_input}")
def text_prediction(text_input):
    
    response = model_text_obj.predict(
        [text_input.split(";")[0],], 
        generator=data_generator,
        for_api=True,
        is_="text"
        )
    return response

@app.get("/api/fusion/predict/text={text_input}&url={image_input:path}")
def fusion_prediction(text_input, image_input):
    
    save_image(image_input)

    response = model_fusion.predict(
        [text_input.split(";")[0],UPLOADED_PATH], 
        generator=data_generator,
        for_api=True,
        model=model_fusion.model,
        is_="fusion"
        )
    return response


