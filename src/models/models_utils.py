from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tqdm.keras import TqdmCallback

def get_tqdm():
    return TqdmCallback(verbose=2)

def get_dashboard(log_dir):
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

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