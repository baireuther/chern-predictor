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
import gc
import json
import os
from time import time as now
from typing import Dict, List

# Third party libraries
import numpy as np
from matplotlib import pyplot as plt

# This project
from chern_predictor.networks.evaluation import accuracy_discrete
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset
from chern_predictor.networks.networks import generate_cnn


def train_ensemble(
    dataset_directory: str,
    num_nets: int,
    network_type: str,
    num_epochs: int,
    ensemble_path: str,
    model_path: str,
    plot_path: str,
    verbose: bool = False,
):
    """Train the individual networks of an ensemble.

    Parameters
    ----------
    dataset_directory: Folder where the datasets are stored
    num_nets: Number of neural networks
    network_type: The (predefined) type of neural networks to be trained. Options are "simplest"
    num_epochs: Number of epochs for which the networks are trained
    ensemble_path: Path where the ensemble information is stored
    model_path: Path where the individual models are stored
    plot_path: Path where the training plots are stored
    verbose: If `True`, some feedback is printed
    """

    # Build the ensemble
    ensemble_name_list = []
    start_time = now()
    for network_index in range(1, num_nets + 1):

        # Generate network name
        network_name = network_type + f"-{network_index}"
        print(f"Training {network_name}...")

        # Bootstrap a training dataset
        data_training, data_validation, data_test = load_data(
            dataset_directory, verbose=False
        )
        data_training_bootstrap = np.random.choice(
            a=data_training, size=len(data_training), replace=True
        ).tolist()
        network_name = train_network(
            dataset_training=data_training_bootstrap,
            dataset_validation=data_validation,
            network_type=network_type,
            num_epochs=num_epochs,
            network_name=network_name,
            model_path=model_path,
            plot_path=plot_path,
            verbose=verbose,
        )
        ensemble_name_list.append(network_name)

        # Delete some unused variables to avoid memory overflow
        del data_training, data_training_bootstrap, data_validation, data_test
        gc.collect()

        if verbose:
            # Print remaining time estimate
            elapsed_time = now() - start_time
            print(
                f"Elapsed time {round(elapsed_time / 60)} minutes, estimated remaining time"
                f" {round(elapsed_time / network_index * (num_nets - network_index) / 60)} minutes.\n"
            )

    with open(ensemble_path + "network_list.json", "w") as file:
        json.dump(ensemble_name_list, file)

    return


def train_network(
    dataset_training: List[Dict],
    dataset_validation: List[Dict],
    network_type: str,
    num_epochs: int,
    network_name: str,
    model_path: str,
    plot_path: str,
    verbose: bool = False,
) -> str:
    """Build and train a neural network.

    Parameters
    ----------
    dataset_training: Training dataset
    dataset_validation: Validation dataset
    network_type: The (predefined) type of neural networks to be trained. Options are
        "cnn32", "cnn64", and "cnn64dl"
    num_epochs: Number of epochs to be trained for
    network_name: A unique name for the neural network
    model_path: Path where the network instances will be stored after each epoch
    plot_path: Path where the training curves are stored
    verbose: If `True`, some output will be printed

    Returns
    -------
    Name of the trained network
    """

    # Prepare the data
    x_training, y_training, y_training_hot = preprocess_dataset(
        dataset_training, normalize=True
    )
    x_validation, y_validation, y_validation_hot = preprocess_dataset(
        dataset_validation, normalize=True
    )

    # Delete some unused variables to avoid memory overflow
    del dataset_training, dataset_validation
    gc.collect()

    # Generate neural network
    num_classes = y_training_hot.shape[-1]
    num_sites_x = x_training.shape[-3]
    num_sites_y = x_training.shape[-2]

    if network_type == "cnn32":
        model = generate_cnn(
            input_shape=(num_sites_x, num_sites_y, 1),
            num_classes=num_classes,
            num_neurons=32,
            num_conv_blocks=2,
            num_conv_layers=1,
            num_fc_layers=1,
            random_flip=True,
            name=network_name,
            verbose=verbose,
        )
    elif network_type == "cnn64":
        model = generate_cnn(
            input_shape=(num_sites_x, num_sites_y, 1),
            num_classes=num_classes,
            num_neurons=64,
            num_conv_blocks=2,
            num_conv_layers=1,
            num_fc_layers=1,
            random_flip=True,
            name=network_name,
            verbose=verbose,
        )
    elif network_type == "cnn64dl":
        model = generate_cnn(
            input_shape=(num_sites_x, num_sites_y, 1),
            num_classes=num_classes,
            num_neurons=64,
            num_conv_blocks=2,
            num_conv_layers=2,
            num_fc_layers=1,
            random_flip=True,
            name=network_name,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Model type {network_type} is not implemented.")

    # Train neural networks
    start_time = now()
    accuracy_dict = {"training": [], "validation": []}
    for epoch in range(num_epochs + 1):
        if epoch == 0:
            if verbose:
                print(f"Evaluating and storing the untrained network.")
        else:
            if verbose:
                print(f"Starting training epoch {epoch}.")
            model.fit(
                x=x_training,
                y=y_training_hot,
                epochs=1,
                batch_size=64,
                verbose=0,
                shuffle=True,
            )

        model.save(
            os.path.join(model_path, network_name + "--epoch-" + str(epoch) + ".h5")
        )

        train_accuracies, _, _ = accuracy_discrete(
            x_training, y_training, model, "Training data", plot=False
        )
        accuracy_dict["training"].append(np.mean(train_accuracies))

        validation_accuracies, _, _ = accuracy_discrete(
            x_validation, y_validation, model, "Validation data", plot=False
        )
        accuracy_dict["validation"].append(np.mean(validation_accuracies))

    with open(
        os.path.join(model_path, network_name + "-prediction-accuracies.json"), "w"
    ) as file:
        json.dump(accuracy_dict, file)

    # Plot training curves
    if plot_path:
        _ = accuracy_discrete(
            x_data=x_training,
            y_data=y_training,
            model=model,
            dataset_name="Training data",
            filename=os.path.join(
                plot_path, network_name + "-training-confusion-matrix.png"
            ),
        )
        _ = accuracy_discrete(
            x_data=x_validation,
            y_data=y_validation,
            model=model,
            dataset_name="Validation data",
            filename=os.path.join(
                plot_path, network_name + "-validation--confusion-matrix.png"
            ),
        )

        plt.plot(
            range(num_epochs + 1),
            np.array(accuracy_dict["training"]),
            "--",
            color="grey",
            label=network_name + " training",
        )
        plt.plot(
            range(num_epochs + 1),
            np.array(accuracy_dict["validation"]),
            "-",
            color="tab:blue",
            label=network_name + " validation",
        )

        plt.legend(loc="lower left", ncol=3)
        plt.ylim(0, 1)
        plt.hlines(
            np.arange(0.3, 1.0, 0.05),
            0.0,
            len(accuracy_dict["validation"]),
            linestyle="--",
        )
        plt.savefig(
            os.path.join(plot_path, network_name + "-training-curves.png"),
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close()

    return network_name
