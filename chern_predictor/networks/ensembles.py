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
import copy
from typing import Tuple

# Third party libraries
import numpy as np
import scipy

# This project
from chern_predictor.networks.evaluation import calc_confusion_matrix


def average_prediction(
    network_predictions: np.ndarray, ground_truth: np.ndarray, minimum_weight: float = 0
) -> Tuple[float, float, np.ndarray]:
    """Calculates the ensemble prediction by averaging over the prediction vectors.

    Parameters
    ----------
    network_predictions: An array with the prediction vectors of the networks constituting the
        ensemble, `shape=(ensemble_size, num_samples, num_labels)`.
    ground_truth: The true labels in the same ordering as in the `network_predictions`,
        `shape=(num_samples, )`.
    minimum_weight: If the weight of an ensemble prediction is below this value, it will be
        discarded.

    Returns
    -------
    prediction accuracy,
    fraction of the predictions that passed the minimum weight threshold,
    confusion matrix
    """
    ensemble_size, num_samples, num_labels = network_predictions.shape

    # Evaluate the ensemble
    ensemble_avg = np.mean(network_predictions, axis=0)
    ensemble_prediction = np.argmax(ensemble_avg, axis=-1)

    # Select those predictions that meet the `minimum_weight` threshold
    selection = np.max(ensemble_avg, axis=-1) >= minimum_weight
    selected_fraction = np.mean(selection)

    prediction_is_success = ensemble_prediction[selection] == ground_truth[selection]
    if prediction_is_success.shape[0] == 0:
        prediction_accuracy = (
            0.0  # Catch the corner case that no prediction passed the threshold
        )
    else:
        prediction_accuracy = np.mean(prediction_is_success, axis=0)

    # Calculate confusion matrix
    confusion_matrix = calc_confusion_matrix(
        ensemble_prediction[selection], ground_truth[selection], num_labels
    )

    return prediction_accuracy, selected_fraction, confusion_matrix


def evaluate_ensemble(
    weights: np.ndarray,
    ground_truth: np.ndarray,
    ensemble_size: int,
    num_bootstraps: int = 1,
    unique_ensembles: bool = False,
    minimum_weight: float = 0.0,
    ensemble_seed: int = None,
) -> Tuple[np.ndarray, float, float, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    weights: `shape=(num_nets, num_samples, num_labels)`
    ground_truth: `shape=(num_samples, )`
    ensemble_size: Number of networks in ensemble
    num_bootstraps: Number of bootstraps
    unique_ensembles: If True, no two ensembles can be the same
    minimum_weight: If the weight of an ensemble prediction is below this value, it will be
        discarded. In case of majority voting the minimum number of votes is calculated from
        this by `min_num_votes = np.round(minimum_weight * ensemble_size)`.
    ensemble_seed: Seed to make the ensembles reproducible

    Returns
    -------
        Prediction accuracies, mean thereof, standard deviation thereof
        Selected fractions of samples, mean thereof, standard deviation thereof
        Mean of confusion matrices
        Standard deviation of confusion matrices
    """

    # A random number generator used in creating the ensembles
    rng = np.random.RandomState(ensemble_seed)

    num_nets, num_data, num_labels = weights.shape
    weights_orig = copy.deepcopy(weights)

    # Normalize the weights properly
    normalization = np.sum(weights, axis=2)
    weights /= np.expand_dims(normalization, 2)

    # Sanity check
    if unique_ensembles:
        if scipy.special.comb(num_nets, ensemble_size) < num_bootstraps:
            raise ValueError(
                "There are not enough unique possibilities to create ensembles."
            )

    # Sample ensembles
    prediction_accuracies = []
    selected_fractions = []
    confusion_matrices = []
    ensemble_history = []
    for _ in range(num_bootstraps):
        unique = False
        while unique == False:
            idcs = sorted(
                rng.choice(list(range(num_nets)), ensemble_size, replace=False)
            )
            if unique_ensembles:
                if idcs not in ensemble_history:
                    unique = True
            else:
                unique = True
        ensemble_history.append(idcs)
        weights = copy.deepcopy(weights_orig)
        network_predictions = weights[idcs]
        (
            prediction_accuracy,
            selected_fraction,
            confusion_matrix,
        ) = average_prediction(
            network_predictions=network_predictions,
            ground_truth=ground_truth,
            minimum_weight=minimum_weight,
        )

        prediction_accuracies.append(prediction_accuracy)
        selected_fractions.append(selected_fraction)
        confusion_matrices.append(confusion_matrix)

    return (
        prediction_accuracies,
        np.mean(prediction_accuracies, axis=0),
        np.std(prediction_accuracies, axis=0),
        selected_fractions,
        np.mean(selected_fractions, axis=0),
        np.std(selected_fractions, axis=0),
        np.mean(confusion_matrices, axis=0),
        np.std(confusion_matrices, axis=0),
    )
