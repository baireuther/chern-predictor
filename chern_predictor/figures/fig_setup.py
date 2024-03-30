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
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

# This project
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset


def gen_setup_fig_data(dataset_path: str, figure_path: str):

    figure_data = {"b": {}, "c": {}}

    _, _, data_test = load_data(
        dataset_path,
        load_training_data=False,
        load_validation_data=False,
        load_test_data=True,
        verbose=False,
    )

    disorder_strengths = []
    for dat in data_test:
        disorder_strengths.append(dat["relative_disorder_strength"])
    selection = np.array(disorder_strengths) == 0.8

    x_test, y_test, _ = preprocess_dataset(
        np.array(data_test)[selection], normalize=True, shuffle=False
    )

    idcs = []
    for chern_number in range(4):
        idcs.append(np.argwhere(y_test == chern_number)[0])

    for chern_number in range(4):
        dat = x_test[idcs[chern_number]].reshape(24, 24)
        figure_data["c"][f"c{chern_number}_data"] = dat.tolist()

    with open(os.path.join(figure_path, "fig_setup_data.json"), "w") as file:
        json.dump(figure_data, file)

    return


def make_setup_fig(figure_path: str):

    with open(os.path.join(figure_path, "fig_setup_data.json"), "r") as file:
        figure_data = json.load(file)

    rc("text", usetex=True)

    fontsize = 28

    fig, axes = plt.subplots(
        1, 5, figsize=(26.83, 4.89), gridspec_kw={"width_ratios": [4, 0.45, 4, 0.1, 4]}
    )
    axes = axes.flatten()

    axes[0].axis("off")
    axes[2].axis("off")

    ax_index = 4
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=axes[ax_index], wspace=-0.1, hspace=0.3
    )
    for chern_number in range(4):
        ax = plt.Subplot(fig, gs[chern_number])
        dat = figure_data["c"][f"c{chern_number}_data"]
        dat = dat[::-1]  # Transpose x-axis to match site ordering in Hamiltonian
        ax.imshow(dat, cmap="Blues", extent=[0.5, 24.5, 0.5, 24.5])
        ax.set_xlabel(r"$x$", fontsize=fontsize)
        ax.set_ylabel(r"$y$", fontsize=fontsize)
        ax.text(x=3.0, y=19.0, s=rf"$|C|={chern_number}$", fontsize=fontsize - 4)
        ax.set_xticks([1, 9, 16, 24])
        ax.set_yticks([1, 9, 16, 24])
        ax.xaxis.set_tick_params(labelsize=fontsize - 8)
        ax.yaxis.set_tick_params(labelsize=fontsize - 8)
        fig.add_subplot(ax)

    axes[1].axis("off")
    axes[3].axis("off")
    axes[4].axis("off")

    # axes[0].text(0, 0.98, r"a)", fontsize=fontsize)
    # axes[2].text(-2.3, 0.75, r"b)", fontsize=fontsize)
    axes[4].text(-0.05, 0.95, r"c)", fontsize=fontsize)

    fig.subplots_adjust(wspace=0.1)
    for ftype in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "setup." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
