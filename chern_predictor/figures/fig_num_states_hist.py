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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# This project
from chern_predictor.data.analysis import extract_data_from_dataset
from chern_predictor.networks.helper_functions import load_data


def gen_number_of_states_histogram_data(
    dataset_directory: str, figure_path: str, verbose: bool = False
):

    figure_data = {}

    rc("text", usetex=True)

    data_train, _, data_test = load_data(
        dataset_directory,
        load_training_data=True,
        load_validation_data=False,
        load_test_data=True,
        verbose=False,
    )

    for chern_number in [0, 1, 2, 3]:
        figure_data[f"chern_number_{chern_number}"] = {}
        for dataset_name, dataset in zip(
            ["Training", "Test"],
            [data_train, data_test],
        ):
            n_evals, _, _, _ = extract_data_from_dataset(
                dataset[:], abs_chern_nums=(chern_number,), verbose=verbose
            )
            figure_data[f"chern_number_{chern_number}"][dataset_name] = {}
            figure_data[f"chern_number_{chern_number}"][dataset_name][
                "n_evals"
            ] = n_evals.tolist()

    with open(
        os.path.join(figure_path, "fig_num_states_in_ldos_data.json"), "w"
    ) as file:
        json.dump(figure_data, file)


def make_number_of_states_histogram(figure_path: str):

    with open(
        os.path.join(figure_path, "fig_num_states_in_ldos_data.json"), "r"
    ) as file:
        figure_data = json.load(file)

    rc("text", usetex=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.95, 6.39), sharey=True, sharex=True)

    axes = axes.flatten()

    for chern_number in [0, 1, 2, 3]:
        ax = axes[chern_number]
        for dataset_name, color, alpha in zip(
            ["Training", "Test"],
            ["tab:blue", "tab:red"],
            [0.55, 0.6],
        ):
            n_evals = np.array(
                figure_data[f"chern_number_{chern_number}"][dataset_name]["n_evals"]
            )
            counts, bins, _ = ax.hist(
                n_evals / 2,
                bins=np.arange(-0.5, 39.5, 1.0),
                weights=np.ones(len(n_evals / 2)) / len(n_evals),
                alpha=alpha,
                color=color,
                label=f"{dataset_name}: {round(np.mean(n_evals) / 2, 1)}",
            )
        ax.set_xlim(-0.5, 29.5)
        ax.locator_params(axis="x", nbins=3)
        legend = ax.legend(
            ncol=1,
            loc="upper right",
            fontsize=14,
            title=rf"$|C|={chern_number}$",
            title_fontsize=16,
            fancybox=False,
        )
        legend._legend_box.align = "left"

    axes[0].set_ylim(0, 0.3)
    axes[0].set_ylabel("count", fontsize=20)
    axes[2].set_ylabel("count", fontsize=20)
    axes[2].set_xlabel("number of states", fontsize=20)
    axes[3].set_xlabel("number of states", fontsize=20)
    axes[0].locator_params(axis="y", nbins=5)
    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    for ax in axes[[1, 2, 3]]:
        ax.set_yticks([])
    axes[0].text(-2.2, 0.22, r"$||$", fontsize=32, rotation=45)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for ftype in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "num_states_in_ldos." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
