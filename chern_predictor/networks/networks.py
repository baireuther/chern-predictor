# Copyright (c) 2020-2023, Paul Baireuther
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with this
# program. If not, see <https://www.gnu.org/licenses/>


# Python standard library
from typing import Tuple

# Third party libraries
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    RandomFlip,
)
from tensorflow.keras.models import Sequential


def generate_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int = 4,
    num_neurons: int = 32,
    num_conv_blocks: int = 1,
    num_conv_layers: int = 1,
    num_fc_layers: int = 1,
    random_flip: bool = False,
    name: str = "cnn",
    verbose: bool = True,
) -> Sequential:
    """Generates a simple sequential CNN model. As loss function `categorical_crossentropy` is
    used and the Adam optimizer is set when compiling the model.

    Parameters
    ----------
    input_shape: Neural network input `shape=(num_sites_x, num_sites_y, num_channels)`
    num_classes: Dimensionality of output
    num_neurons: Number of neurons in each layer
    num_conv_blocks: Number of conv blocks
    num_conv_layers: Number of conv layers in each conv block
    num_fc_layers: Number of fully connected hidden layers after last conv block
    random_flip: If `True`, randomly flip the input 'image' for data augmentation
    name: Name of the neural network
    verbose: If `True`, some info about the network is printed

    Returns
    -------
    Compiled Keras model
    """

    if len(input_shape) != 3:
        raise ValueError(
            f"Input shape must be (num_sites_x, num_sites_y, num_channels)."
        )

    model = Sequential(name=name)

    model.add(Input(input_shape))

    if random_flip:
        model.add(RandomFlip())

    for _ in range(num_conv_blocks):
        for _ in range(num_conv_layers):
            model.add(
                Conv2D(
                    num_neurons,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                )
            )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

    model.add(Flatten())

    for _ in range(num_fc_layers):
        model.add(Dense(num_neurons, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if verbose:
        print(model.summary())

    return model
