"""Canonical model architecture shared by training and aggregation."""
import os
import warnings

import keras
import keras.layers as tfkl

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    import tensorflow_probability as tfp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ARCHITECTURE = {
    "cnn_layers": 7,
    "filters": [16, 16, 32, 32, 32, 16, 16],
    "kernel_size": [5, 5, 3, 3, 3, 3, 3],
    "pooling": [1, 2, 2, 2, 2, 2, 2],
    "dense_layers": 2,
    "neurons_dense": [32, 16],
}

HYPER_PARAMETERS = {
    "epochs": 105,
    "lr": 0.007,
    "amp_sdo_c": 0.15,
    "amp_do_d": 0.15,
    "amp_gn": 0.03,
    "rot": 0.1,
    "batch_size": 600,
}


class RotationLayer(tfkl.Layer):
    def __init__(self, intensity, **kwargs):
        self.intensity = intensity
        super().__init__(**kwargs)

    def call(self, x, training=None):
        def noised():
            batch = tf.shape(x)[0]
            randnormal = tf.random.normal((batch, 3, 3))
            qrinput = randnormal * self.intensity + (1.0 - self.intensity) * tf.eye(3)
            q = tfp.math.gram_schmidt(qrinput)
            return tf.transpose(tf.matmul(q, x, transpose_b=True), [0, 2, 1])
        return noised() if training else x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "intensity": self.intensity}


def _block(filters, kernel_size, spatial_dropout, max_pool, gaussian_noise, batchnorm_axis, mo):
    initializer = tf.keras.initializers.HeUniform()
    if gaussian_noise > 0:
        mo = tfkl.GaussianNoise(gaussian_noise)(mo)
    mo = tfkl.Conv1D(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer, padding="same")(mo)
    if batchnorm_axis is not None:
        mo = tfkl.BatchNormalization(axis=batchnorm_axis)(mo)
    mo = tfkl.Activation("swish")(mo)
    if spatial_dropout > 0:
        mo = tfkl.SpatialDropout1D(spatial_dropout)(mo)
    if max_pool > 1:
        mo = tfkl.MaxPool1D(max_pool)(mo)
    return mo


def _dense_block(no_neurons, drop_out, batch_norm, mo):
    initializer = tf.keras.initializers.HeUniform()
    mo = tfkl.Dense(no_neurons, kernel_initializer=initializer)(mo)
    if batch_norm:
        mo = tfkl.BatchNormalization()(mo)
    mo = tfkl.ReLU()(mo)
    return tfkl.Dropout(drop_out)(mo)


def load_model(params: dict = HYPER_PARAMETERS, architecture: dict = ARCHITECTURE,
                t_dimension: int = 192, no_classes: int = 5, optimizer: str = "adam",
                only_inference: bool = False) -> keras.models.Model:
    """Build and compile the protego-FL CNN classifier."""
    rot, lr = params["rot"], params["lr"]

    assert architecture["cnn_layers"] == len(architecture["filters"])
    assert architecture["cnn_layers"] == len(architecture["kernel_size"])
    assert architecture["cnn_layers"] == len(architecture["pooling"])
    assert architecture["dense_layers"] == len(architecture["neurons_dense"])

    model_input = tfkl.Input(shape=(t_dimension, 3), name="input")
    mo = model_input
    if not only_inference:
        mo = RotationLayer(rot)(mo)

    for ix in range(architecture["cnn_layers"]):
        sp_do = 0 if ix == 0 else params["amp_sdo_c"]
        gn_c = params["amp_gn"] if ix <= 1 else 0
        mo = _block(
            filters=architecture["filters"][ix],
            kernel_size=architecture["kernel_size"][ix],
            spatial_dropout=sp_do,
            max_pool=architecture["pooling"][ix],
            gaussian_noise=gn_c,
            batchnorm_axis=-1,
            mo=mo,
        )

    mo = tfkl.GlobalAveragePooling1D()(mo)
    for ix in range(architecture["dense_layers"]):
        mo = _dense_block(architecture["neurons_dense"][ix], params["amp_do_d"], batch_norm=True, mo=mo)

    mo = tfkl.Softmax()(tfkl.Dense(no_classes)(mo))
    model = keras.models.Model(inputs=model_input, outputs=mo)

    if not only_inference:
        metrics = [
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.F1Score(name="f1_weighted", average="weighted"),
        ]
        if optimizer.lower() == "adam":
            opt = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer.lower() == "sgd":
            opt = keras.optimizers.SGD(learning_rate=lr)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            opt = optimizer
        else:
            raise ValueError("Unknown optimizer")
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=metrics)

    return model
