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
from typing import Optional, Tuple

# Third party libraries
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential

# This project
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset


def evaluate_networks_in_ensemble(
    dataset_directory: str, ensemble_path: str, model_path: str, verbose: bool = False
):
    """Evaluate all networks in an ensemble individually on the training, validation, and test
    dataset.

    Parameters
    ----------
    dataset_directory: The folder in which the datasets are stored
    ensemble_path: The folder in which the ensembles are stored
    model_path: The folder in which the individual neural network models are stored
    verbose: If `True`, some feedback is printed
    """

    # Load datasets and prepare data
    data_training, data_validation, data_test = load_data(dataset_directory)

    x_training, y_training, _ = preprocess_dataset(data_training, normalize=True)
    x_validation, y_validation, _ = preprocess_dataset(data_validation, normalize=True)
    x_test, y_test, _ = preprocess_dataset(data_test, normalize=True)
    datasets = {
        "training": (x_training, y_training),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }

    # Delete some unused variables to avoid memory overflow
    del data_training, data_validation, data_test
    gc.collect()

    # Get names of networks in ensemble
    with open(ensemble_path + "network_list.json", "r") as file:
        model_name_list = json.load(file)

    # Evaluate the networks
    start_time = now()
    num_models = len(model_name_list)
    for model_index in range(1, num_models + 1):
        model_name = model_name_list[model_index - 1]
        print(f"Evaluating {model_name}.")

        # Dictionary to store results
        results = {"training": None, "validation": None, "test": None}
        for key in results:
            results[key] = {
                "mean_accuracies": [],
                "accuracies": [],
                "confusion_matrices": [],
                "prediction_vector": [],
            }

        # Iterate over all training epochs and evaluate the model
        epoch = 0
        while os.path.exists(
            os.path.join(model_path, model_name + f"--epoch-{str(epoch)}.h5")
        ):
            model = load_model(
                os.path.join(model_path, model_name + f"--epoch-{str(epoch)}.h5")
            )
            for key in results:
                x, y = datasets[key]
                accuracies, confusion_matrices, prediction_vectors = accuracy_discrete(
                    x_data=x,
                    y_data=y,
                    model=model,
                    dataset_name=key,
                    plot=False,
                    return_prediction_vector=True,
                )
                results[key]["mean_accuracies"].append(np.mean(accuracies))
                results[key]["accuracies"].append(accuracies.tolist())
                results[key]["confusion_matrices"].append(confusion_matrices.tolist())
                results[key]["prediction_vector"].append(prediction_vectors.tolist())
            epoch += 1

        if verbose:
            # Print remaining time estimate
            elapsed_time = now() - start_time
            print(
                f"Elapsed time {round(elapsed_time / 60)} minutes, estimated remaining time"
                f" {round(elapsed_time / model_index * (num_models - model_index) / 60)} minutes.\n"
            )

        with open(
            os.path.join(model_path, model_name + "-evaluated.json"), "w"
        ) as file:
            json.dump(results, file)

        # Delete some unused variables to avoid memory overflow
        del results
        gc.collect()

    return


def accuracy_discrete(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model: Sequential,
    dataset_name: str = "",
    plot: bool = True,
    filename: Optional[str] = None,
    return_prediction_vector: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluate the accuracy of a neural network.

    Parameters
    ----------
    x_data: Input values
    y_data: Ground truth
    model: Neural network model
    dataset_name: Name of the dataset
    plot: If true, confusion matrices will be plotted
    filename: If provided the confusion matrix plot will be stored on disc under this filename
    return_prediction_vector: If true, the prediction vectors are returned
    verbose: If true, some feedback is printed

    Returns
    -------
    prediction accuracies
    confusion matrix
    prediction vectors IF `return_prediction_vector == True`
    """

    # Calculate predictions
    y_prediction_vec = model.predict(x_data, verbose=verbose)
    _, num_classes = y_prediction_vec.shape
    y_predicted = np.argmax(y_prediction_vec, axis=-1)
    if verbose:
        average_prediction_accuracy = np.mean(y_predicted == y_data)
        print(
            f"Model {model.name} achieved and average prediction accuracy of "
            f"{average_prediction_accuracy} on the {dataset_name} dataset."
        )

    # Calculate confusion matrix
    confusion_matrix = calc_confusion_matrix(
        prediction=y_predicted, ground_truth=y_data, num_classes=num_classes
    )

    # Calculate accuracies resolved by class
    prediction_accuracy_by_class = np.array(
        [confusion_matrix[i][i] for i in range(num_classes)]
    )
    if verbose:
        print(
            f"Model {model.name} achieved an class resolved prediction accuracy of "
            f"{round(prediction_accuracy_by_class, 2)} on the {dataset_name} dataset."
        )

    if plot:
        plot_confusion_matrix(
            confusion_matrix,
            title=f"Accuracies = {np.round(np.mean(prediction_accuracy_by_class), 2)}",
            fname=filename,
        )

    if return_prediction_vector:
        # Delete some unused variables to avoid memory overflow
        del y_predicted
        gc.collect()
        return prediction_accuracy_by_class, confusion_matrix, y_prediction_vec
    else:
        # Delete some unused variables to avoid memory overflow
        del y_prediction_vec, y_predicted
        gc.collect()
        return prediction_accuracy_by_class, confusion_matrix, None


def calc_confusion_matrix(
    prediction: np.ndarray, ground_truth: np.ndarray, num_classes: int
) -> np.ndarray:
    """Calculates the confusion matrix.

    Parameters
    ----------
    prediction: Predicted labels, `shape=(n_samples, )`
    ground_truth: True labels, `shape=(n_samples, )`
    num_classes: Number of classes

    Returns
    -------
    Confusion matrix, `shape=(n_samples, n_samples)`
    """
    n_samples = ground_truth.shape[0]
    if n_samples == 0:
        # In the corner case, that there is no prediction, maximal confusion is assumed and all
        # values in the confusion matrix are the same.
        confusion_matrix = np.ones(shape=(num_classes, num_classes)) / num_classes
    else:
        confusion_tensor = np.zeros(shape=(num_classes * num_classes, n_samples))
        tensor_elements = prediction * num_classes + ground_truth
        confusion_tensor[tensor_elements, np.arange(tensor_elements.size)] = 1
        confusion_matrix = np.mean(confusion_tensor, axis=1).reshape(
            num_classes, num_classes
        )
        confusion_matrix /= np.expand_dims(np.sum(confusion_matrix, axis=0), 0) + 1e-12
    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = "",
    fname: Optional[str] = None,
):
    """Plot a confusion matrix."""

    fig, ax12 = plt.subplots(1, 1, figsize=(5, 5))
    c = ax12.imshow(confusion_matrix, vmin=vmin, vmax=vmax, origin="lower")
    ax12.set_aspect("equal", "box")
    ax12.set_title(title, fontsize=20)
    fig.colorbar(c, ax=ax12)
    for ax in [ax12]:
        ax.set_xlabel("true", fontsize=16)
        ax.set_ylabel("predicted", fontsize=16)
        for tl in ax.get_xticklabels():
            tl.set_color("k")
            tl.set_size(16)
        for tl in ax.get_yticklabels():
            tl.set_color("k")
            tl.set_size(16)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)
        plt.close()
    else:
        plt.show()

    return
