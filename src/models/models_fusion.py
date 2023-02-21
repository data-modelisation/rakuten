
from tensorflow.keras.layers import concatenate, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def model_multi_input(model_text, model_image):

    combined = concatenate([model_text.output, model_image.output])

    z = Dense(54, activation="relu")(combined)
    z = Dense(27, activation="softmax")(z)
    
    model = Model(inputs=[model_text.input, model_image.input], outputs=z)
    #.layers["class"]
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
