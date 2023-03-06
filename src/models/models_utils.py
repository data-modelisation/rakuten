import keras
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tqdm.keras import TqdmCallback
#import resource
import gc

class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)

METRICS = [
    'accuracy',
    tf.keras.metrics.SparseCategoricalAccuracy(),
    #BalancedSparseCategoricalAccuracy(),
]


class memoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        gc.collect()
        keras.backend.clear_session()

def call_memory():
    return memoryCallback()

def call_tqdm():
    return TqdmCallback(verbose=2)

def call_dashboard(log_dir):
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

def call_checkpoint(path, monitor="accuracy"):
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

def call_earlystopping():
    return EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        restore_best_weights=True,
        patience=5, 
        verbose=2, 
        mode='min')

def call_reducelr():
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=2,
        mode='auto',
        min_delta=0.0001,
        cooldown=2,
    ) 