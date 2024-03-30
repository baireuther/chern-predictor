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

# Third party libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

# This project
from chern_predictor.networks.ensembles import evaluate_ensemble
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset
from default_parameters import evaluation_params, network_parameters


def gen_threshold_plot_data(
    ensemble_path: str,
    ensemble_seed: int,
    model_path: str,
    dataset_directory: str,
    figure_path: str,
    verbose: bool = False,
):

    data = {}

    _, _, data_test = load_data(
        dataset_directory,
        load_training_data=False,
        load_validation_data=False,
        load_test_data=True,
        verbose=False,
    )
    _, y_test, _ = preprocess_dataset(data_test, normalize=True)

    with open(ensemble_path + "network_list.json", "r") as file:
        model_name_list = json.load(file)

    data_name = "test"
    weights_test = []
    for model_name in model_name_list:
        with open(
            os.path.join(model_path, model_name + "-evaluated.json"), "r"
        ) as file:
            accs_dict = json.load(file)
        w = accs_dict[data_name]["prediction_vector"]
        weights_test.append(w)
    weights_test = np.array(weights_test)

    n_bootstraps = evaluation_params["num_bootstraps"]
    n_epochs_min = evaluation_params["epochs_until_trained"] + 1
    n_epochs = weights_test.shape[1]
    n_steps = 49
    delta = 1 / (n_steps + 1)
    ensemble_sizes = evaluation_params["ensemble_sizes"]

    res = np.zeros((len(ensemble_sizes), n_steps, 5))
    for m, ensemble_size in enumerate(ensemble_sizes):
        data[f"ensemble_size_{ensemble_size}"] = {}

        if verbose:
            print(f"ensemble size = {ensemble_size}")
        for step, threshold in enumerate(np.linspace(delta, 1.0 - delta, n_steps)):
            accs = []
            fracs = []
            for epoch in range(n_epochs_min, n_epochs):
                acc, acc_mean, acc_std, frac, _, _, cm_mean, _ = evaluate_ensemble(
                    weights=weights_test[:, epoch, :, :],
                    ground_truth=y_test,
                    ensemble_size=ensemble_size,
                    num_bootstraps=n_bootstraps,
                    unique_ensembles=True,
                    minimum_weight=threshold,
                    ensemble_seed=ensemble_seed + m,
                )
                accs.append(acc)
                fracs.append(frac)
            mean_acc = np.mean(np.array(accs))
            std_acc = np.std(np.array(accs))
            mean_fraction = np.mean(np.array(fracs))
            std_fraction = np.std(np.array(fracs))
            res[m, step, 0] = threshold
            res[m, step, 1] = mean_acc
            res[m, step, 2] = std_acc
            res[m, step, 3] = mean_fraction
            res[m, step, 4] = std_fraction

        data[f"ensemble_size_{ensemble_size}"]["threshold"] = res[m, :, 0].tolist()
        data[f"ensemble_size_{ensemble_size}"]["mean_acc"] = res[m, :, 1].tolist()
        data[f"ensemble_size_{ensemble_size}"]["std_acc"] = res[m, :, 2].tolist()
        data[f"ensemble_size_{ensemble_size}"]["mean_fraction"] = res[m, :, 3].tolist()
        data[f"ensemble_size_{ensemble_size}"]["std_fraction"] = res[m, :, 4].tolist()

    with open(
        os.path.join(figure_path, "fig_certainty_threshold_data.json"), "w"
    ) as file:
        json.dump(data, file)


def make_threshold_figure(
    ax_accuracy,
    ax_fraction,
    figure_path: str,
):
    # Load data
    with open(
        os.path.join(figure_path, "fig_certainty_threshold_data.json"), "r"
    ) as file:
        figure_data = json.load(file)

    # Config
    rc("text", usetex=True)
    fontsize = 28
    colors = evaluation_params["colors"]

    for m, ensemble_size in enumerate(evaluation_params["ensemble_sizes"]):

        threshold = np.array(figure_data[f"ensemble_size_{ensemble_size}"]["threshold"])
        mean_acc = np.array(figure_data[f"ensemble_size_{ensemble_size}"]["mean_acc"])
        std_acc = np.array(figure_data[f"ensemble_size_{ensemble_size}"]["std_acc"])
        mean_fraction = np.array(
            figure_data[f"ensemble_size_{ensemble_size}"]["mean_fraction"]
        )
        std_fraction = np.array(
            figure_data[f"ensemble_size_{ensemble_size}"]["std_fraction"]
        )

        label = f"n={ensemble_size}"
        ax_accuracy.plot(threshold, mean_acc, "-", label=label, color=colors[m], lw=2)
        ax_accuracy.fill_between(
            threshold,
            mean_acc - std_acc,
            mean_acc + std_acc,
            color=colors[m],
            alpha=0.2,
        )
        ax_fraction.plot(
            threshold, mean_fraction, "-", label=label, color=colors[m], lw=2
        )
        ax_fraction.fill_between(
            threshold,
            mean_fraction - std_fraction,
            mean_fraction + std_fraction,
            color=colors[m],
            alpha=0.2,
        )

    ax_accuracy.set_ylabel("prediction accuracy", fontsize=fontsize)
    ax_accuracy.set_ylim(0.9, 1.0)
    ax_accuracy.locator_params(axis="y", nbins=3)
    ax_fraction.set_ylim(0.6, 1.0)
    ax_fraction.set_ylabel(
        "fraction of samples", fontsize=fontsize, rotation=270, labelpad=24
    )
    ax_fraction.yaxis.set_label_position("right")
    ax_fraction.yaxis.tick_right()
    ax_fraction.locator_params(axis="y", nbins=5)
    ax_fraction.legend(loc="lower left", fontsize=fontsize - 4, ncol=1)

    for ax in [ax_accuracy, ax_fraction]:
        ax.set_xlim(0, 1)
        ax.xaxis.set_tick_params(labelsize=fontsize - 4)
        ax.yaxis.set_tick_params(labelsize=fontsize - 4)
        ax.set_xlabel("threshold", fontsize=fontsize)
        ax.locator_params(axis="x", nbins=6)

    ax_accuracy.hlines([0.95, 0.98, 0.99], 0, 1, linestyle="--", color="gray")
    ax_fraction.hlines(
        np.linspace(0, 1, 11),
        0,
        network_parameters["num_epochs"],
        linestyle="--",
        color="gray",
    )


def make_certainty_threshold_fig(
    figure_path: str,
):

    # Config
    fontsize = 28
    rc("text", usetex=True)

    # Make figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    axes = axes.flatten()

    # Make threshold figure
    make_threshold_figure(
        ax_accuracy=axes[0], ax_fraction=axes[1], figure_path=figure_path
    )

    # Misc
    axes[0].text(-0.3, 0.995, r"a)", fontsize=fontsize)
    axes[1].text(-0.12, 0.983, r"b)", fontsize=fontsize)

    fig.subplots_adjust(wspace=0.3)
    for ftype in ["png", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "certainty_threshold." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
