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
from default_parameters import evaluation_params, network_parameters


def gen_training_curve_data(ensemble_path: str, model_path: str, figure_path: str):

    figure_data = {}

    with open(ensemble_path + "network_list.json", "r") as file:
        model_name_list = json.load(file)
    model_name = model_name_list[0]

    with open(os.path.join(model_path, model_name + "-evaluated.json"), "r") as file:
        accs_dict = json.load(file)

    figure_data["accs_training"] = accs_dict["training"]["mean_accuracies"]
    figure_data["accs_validation"] = accs_dict["validation"]["mean_accuracies"]
    figure_data["accs_test"] = accs_dict["test"]["mean_accuracies"]

    with open(
        os.path.join(figure_path, "fig_training_curve_example_data.json"), "w"
    ) as file:
        json.dump(figure_data, file)


def make_training_curve_fig(figure_path: str):

    with open(
        os.path.join(figure_path, "fig_training_curve_example_data.json"), "r"
    ) as file:
        figure_data = json.load(file)

    accs_training = np.array(figure_data["accs_training"])
    accs_validation = np.array(figure_data["accs_validation"])
    accs_test = np.array(figure_data["accs_test"])

    rc("text", usetex=True)

    # Config
    fontsize = 23
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:red"]
    n_epochs = len(accs_training) - 1

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.48))
    axes = [ax]

    axes[0].plot(
        range(n_epochs + 1), accs_training, lw=1.5, label="Training", color=colors[0]
    )
    axes[0].plot(
        range(n_epochs + 1),
        accs_validation,
        lw=1.5,
        label="Validation",
        color=colors[1],
    )
    axes[0].plot(range(n_epochs + 1), accs_test, lw=1.5, label="Test", color=colors[3])

    for ax in axes:
        ax.hlines(
            np.linspace(0, 1, 21),
            0,
            network_parameters["num_epochs"],
            linestyle="--",
            color="gray",
        )
        ax.set_xlim(0, network_parameters["num_epochs"])
        ax.set_ylim(0.85, 1)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize - 4)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize - 4)
    axes[0].set_xlabel("training epoch", fontsize=fontsize)
    axes[0].set_ylabel("prediction accuracy", fontsize=fontsize)
    axes[0].vlines(
        evaluation_params["epochs_until_trained"] + 0.5,
        0,
        1,
        linestyle="dashed",
        color="gray",
    )
    axes[0].locator_params(axis="y", nbins=4)
    axes[0].legend(ncol=2, loc="lower left", fontsize=fontsize - 7)

    # Misc
    fig.subplots_adjust(wspace=0)

    for ftype in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "training_curve_example." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
