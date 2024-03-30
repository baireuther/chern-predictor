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
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib import rc

# This project
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset


def gen_setup_fig_data(dataset_path: str, figure_path: str, verbose=False):

    figure_data = {"b": {}, "c": {}}

    _, _, data_test = load_data(
        dataset_path,
        load_training_data=False,
        load_validation_data=False,
        load_test_data=True,
        verbose=False,
    )

    data_path = "chern_predictor/figures/data/"
    phase_diagram_fname = "phase_diagram_gap_and_kf_in_0.1_12_e0_in_-0.8_0.8.txt"
    phase_diagram = np.loadtxt(data_path + phase_diagram_fname)
    if verbose:
        print(
            f"Fraction of outliers in phase diagram data is:"
            f" {np.mean(np.abs(phase_diagram[:, 5]) > 20)}"
        )

    # Select window
    kf = phase_diagram[:, 0]
    kf_selection = np.abs(kf - 7.5 / 2) < 7.5 / 2
    phase_diagram = phase_diagram[kf_selection]
    kf = phase_diagram[:, 0]
    e0 = phase_diagram[:, 1]
    n_kf = len(set(kf))
    n_e0 = len(set(e0))
    dat = np.abs(phase_diagram[:, 5]).reshape(n_kf, n_e0).T

    figure_data["b"]["data"] = dat.tolist()
    figure_data["b"]["c_max"] = 4
    figure_data["b"]["kf"] = kf.tolist()
    figure_data["b"]["n_kf"] = n_kf
    figure_data["b"]["e0"] = e0.tolist()
    figure_data["b"]["n_e0"] = n_e0

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

    ax_index = 2
    cmap = plt.cm.get_cmap("Blues", figure_data["b"]["c_max"] + 1)
    img = axes[2].imshow(
        figure_data["b"]["data"],
        extent=[
            np.min(figure_data["b"]["kf"]),
            np.max(figure_data["b"]["kf"]),
            np.min(figure_data["b"]["e0"]),
            np.max(figure_data["b"]["e0"]),
        ],
        vmin=-0.5,
        vmax=figure_data["b"]["c_max"] + 0.5,
        cmap=cmap,
        aspect="auto",
        origin="lower",
    )

    conture = patches.Rectangle(
        xy=(2.508, -0.5),
        width=2.492,
        height=1.2,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )
    axes[ax_index].add_patch(conture)
    axes[ax_index].set_xlim(0, 7.5)
    axes[ax_index].set_ylim(-0.8, 0.8)
    axes[ax_index].set_xlabel(r"$k_F \xi$", fontsize=fontsize)
    axes[ax_index].set_ylabel(r"$\epsilon_0 / \Delta_0$", fontsize=fontsize)
    cbar = fig.colorbar(img, ax=axes[ax_index])
    cbar.ax.tick_params(labelsize=fontsize - 4)
    cbar.set_label(r"Chern number $|C|$", fontsize=fontsize, labelpad=0)

    for ax in axes[[ax_index]]:
        ax.xaxis.set_tick_params(labelsize=fontsize - 4)
        ax.yaxis.set_tick_params(labelsize=fontsize - 4)
    axes[ax_index].locator_params(axis="x", nbins=8)
    axes[ax_index].locator_params(axis="y", nbins=9)

    fig.canvas.draw()
    yticks_loc = cbar.ax.get_yticks().tolist()
    cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(yticks_loc))
    ylabels = [item.get_text() for item in cbar.ax.get_yticklabels()]
    ylabels[5] = "$\\mathdefault{\geq 4}$"
    cbar.ax.set_yticklabels(ylabels)

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
    axes[2].text(-2.3, 0.75, r"b)", fontsize=fontsize)
    axes[4].text(-0.05, 0.95, r"c)", fontsize=fontsize)

    fig.subplots_adjust(wspace=0.1)
    for ftype in ["png", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "setup." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
