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
import json
import os
from typing import Dict, List, Tuple

# Third party libraries
import numpy as np


def preprocess_dataset(
    dataset: list,
    normalize: bool = False,
    allow_zero_ldos: bool = True,
    shuffle: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform data such that it can be fed into a neural network.

    Parameters
    ----------
    dataset: The dataset
    normalize: Normalize each LDOS to have unit standard deviation (means are not shifted)
    allow_zero_ldos: If `True`, datapoints with no states in the LDOS will also be returned
    shuffle: If `True`, the dataset will be shuffled

    Returns
    -------
    inputs
    labels
    one-hot labels
    """

    # Infer number of classes from data
    abs_chern_numbers = []
    for datapoint in dataset:
        abs_chern_numbers.append(datapoint["chern_number_absolute_value"])
    unique_values = list(set(abs_chern_numbers))
    n_classes = len(unique_values)

    # Sanity check
    if np.min(unique_values) != 0 or np.max(unique_values) != len(unique_values) - 1:
        raise ValueError(
            "It is assumed that the modulus of the Chern number attains the values "
            "C=0, 1, ..., C_max."
        )

    # Extract the data
    inputs, labels, one_hot_labels = [], [], []
    for datapoint in dataset:
        if allow_zero_ldos or datapoint["num_states_in_ldos"] > 0:
            abs_chern_number = datapoint["chern_number_absolute_value"]
            one_hot_label = np.zeros(n_classes, dtype=bool)
            one_hot_label[abs_chern_number] = True
            inputs.append(datapoint["local_density_of_states"])
            labels.append(abs_chern_number)
            one_hot_labels.append(one_hot_label)

    # Format and optionally shuffle the data
    idcs = np.array(range(len(labels)))
    if shuffle:
        np.random.shuffle(idcs)

    num_sites_x = dataset[0]["system_params"]["num_sites_x"]
    num_sites_y = dataset[0]["system_params"]["num_sites_y"]
    inputs = np.array(inputs)[idcs].reshape(-1, num_sites_x, num_sites_y)
    labels = np.array(labels)[idcs]
    one_hot_labels = np.array(one_hot_labels)[idcs]

    # Optionally, normalize the data; Note that only the scale is normalized but the mean is not
    # shifted.
    if normalize:
        inputs = inputs.reshape(-1, num_sites_x * num_sites_y)
        inputs /= (
            np.std(inputs, axis=-1).reshape(-1, 1) + 1e-9
        )  # To avoid division by zero 1e-9 is added
        inputs = inputs.reshape(-1, num_sites_x, num_sites_y, 1)

    return inputs, labels, one_hot_labels


def load_data(
    dataset_directory: str,
    load_training_data=True,
    load_validation_data=True,
    load_test_data=True,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    if load_training_data:
        with open(os.path.join(dataset_directory, "training.json"), "r") as file:
            data_training = json.load(file)
        if verbose:
            print("Number of training datapoints is {}.".format(len(data_training)))
    else:
        data_training = None

    if load_validation_data:
        with open(os.path.join(dataset_directory, "validation.json"), "r") as file:
            data_validation = json.load(file)
        if verbose:
            print("Number of validation datapoints is {}.".format(len(data_validation)))
    else:
        data_validation = None

    if load_test_data:
        with open(os.path.join(dataset_directory, "test.json"), "r") as file:
            data_test = json.load(file)
        if verbose:
            print("Number of test datapoints is {}.\n".format(len(data_test)))
    else:
        data_test = None

    return data_training, data_validation, data_test


def prepare_folders(
    experiment_directory: str, ensemble_name: str, verbose: bool = True
) -> Tuple[str, str, str, str, str]:
    """Prepare and if not existent, create folders."""

    dataset_path = os.path.join(experiment_directory, "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    ensemble_path = os.path.join(experiment_directory, "ensembles/", ensemble_name)
    if not os.path.exists(ensemble_path):
        os.makedirs(ensemble_path)
    model_path = os.path.join(ensemble_path, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    plot_path = os.path.join(ensemble_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    figure_path = os.path.join(ensemble_path, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    if verbose:
        print(f"The datasets will be stored in {dataset_path}.")
        print(f"The ensemble will be stored in {ensemble_path}.")
        print(f"The individual network models will be stored in {model_path}.")
        print(f"The plots will be stored in {plot_path}.")
        print(f"The figures will be stored in {figure_path}.\n\n")

    return dataset_path, ensemble_path, model_path, plot_path, figure_path
