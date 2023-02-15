from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense


def flow_image_generators(Image_train, Image_test):
    train_generator = Image_train.flow_from_dataframe(
            dataframe=df_train_image,
            x_col="links",
            y_col="label",
            shear_range=.1,
            rotation_range=5,
            zoom_range=.1,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip=True,
            target_size=target_shape,
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=False)

def cnn_simple(input_shape):
    model = Sequential()

    model.add(Conv2D(
            filters = 32, 
            kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape =  input_shape,  
            padding = 'same',
            ))

    model.add(MaxPooling2D(
            pool_size = (3, 3),
            padding = 'same'))

    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(units=27, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model