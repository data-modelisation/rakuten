from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Cropping2D


def flow_image_generators(Image_train, Image_test):
    train_generator = Image_train.flow_from_dataframe(
            dataframe=df_train_image,
            x_col="links",
            y_col="label",
        #     shear_range=.1,
        #     rotation_range=5,
        #     zoom_range=[.1,.1]
        #     width_shift_range = 0.1,
        #     height_shift_range = 0.1,
        #     horizontal_flip=True,
            target_size=target_shape,
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=False)

def mo_cnn_basic(input_shape):
        
        model = Sequential()

        crop_pixels = 10

        croped_shaped = (input_shape[0]-2*crop_pixels, input_shape[1]-2*crop_pixels, input_shape[2])

        model.add(Cropping2D(cropping=((crop_pixels, crop_pixels), (crop_pixels, crop_pixels))))
        
        model.add(Conv2D(
            filters = 16, 
            kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape = croped_shaped,  
            padding = 'valid',
            ))

        model.add(MaxPooling2D(
            pool_size = (2, 2),
            padding = 'valid'))

        model.add(Dropout(.2))

        model.add(Conv2D(
            filters = 8, 
            kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape =  input_shape,  
            padding = 'valid',
            ))

        model.add(MaxPooling2D(
            pool_size = (2, 2),
            padding = 'valid'))

        model.add(Dropout(.2))

        model.add(Flatten())
        model.add(Dense(units=27, activation="softmax"))

        model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
        )

        return model
    

def cnn_simple(input_shape):
        model = Sequential()

        crop_pixels = 10

        croped_shaped = (input_shape[0]-2*crop_pixels, input_shape[1]-2*crop_pixels, input_shape[2])

        model.add(Cropping2D(cropping=((crop_pixels, crop_pixels), (crop_pixels, crop_pixels))))
        
        model.add(Conv2D(
            filters = 16, 
            kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape = croped_shaped,  
            padding = 'valid',
            ))

        model.add(MaxPooling2D(
            pool_size = (2, 2),
            padding = 'valid'))

        model.add(Dropout(.2))

        model.add(Conv2D(
            filters = 8, 
            kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape =  input_shape,  
            padding = 'valid',
            ))

        model.add(MaxPooling2D(
            pool_size = (2, 2),
            padding = 'valid'))

        model.add(Dropout(.2))

        model.add(Flatten())
        model.add(Dense(units=27, activation="softmax"))

        model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
        )

        return model