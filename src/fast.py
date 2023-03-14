from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests

from generators.generator import DataGenerator
from models.models_text import *
from models.models_image import *
from models.models_fusion import *

UPLOADED_PATH = 'uploaded_image.jpg'

#Instance
app = FastAPI()

#Origins
origins = ["https://localhost:8000",]


#Modeles 
model_text = ModelText_Neural_Simple(
        suffix="",
        load=True,
    )

model_image = ModelImage_MobileNet(
        suffix="_224",
        load=True,
    )
    
model_fusion = ModelFusion(
        suffix="_mobilenet_simple_224",
        load=True,
    )

#DataGenrator
data_generator = DataGenerator(
    from_api=True,
    target_shape=(224,224,3)
)

def save_image(url):
    img_data = requests.get(str(url)).content
    with open(UPLOADED_PATH, 'wb') as handler:
        handler.write(img_data)

@app.get("/api/image/layer/{idx_layer}")
def get_layer(idx_layer):
    model_text = ModelText_Neural_Simple(
        suffix="_translated",
        load=True,
    )
    layers = model_text.model.layers

    return StreamingResponse(image.read(), media_type="image/jpeg")


@app.get("/api/image/predict/{image_input:path}")
def pred_image(image_input: str):
    
    save_image(image_input)

    response = model_image.predict(
        [UPLOADED_PATH,], 
        generator=data_generator,
        #model=model_image.model,
        for_api=True,
        is_="image"
    )
    return response

@app.get("/api/text/predict/{text_input}")
def text_prediction(text_input):
    
    response = model_text.predict(
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
        is_="fusion"
        )
    return response


