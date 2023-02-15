from tensorflow.keras.callbacks import ModelCheckpoint

def get_model_checkpoint(path, monitor="accuracy"):
    return ModelCheckpoint(
        path,
        monitor = monitor,
        verbose = 0,
        save_best_only = True,
        save_weights_only= False,
        mode="auto",
        save_freq="epoch",
        options=None,
        initial_value_threshold=None,
    )