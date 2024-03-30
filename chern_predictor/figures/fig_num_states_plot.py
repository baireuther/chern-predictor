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


def gen_number_of_states_plot_data(
    dataset_directory: str, figure_path: str, verbose: bool = False
):
    """The model used to generate the data has particle-hole symmetry. Therefore, in
    previous versions of the code, only half the states in a symmetric energy window
    around zero were considered."""
    figure_data = {}

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
            (n_evals, _, _, u0_factors) = extract_data_from_dataset(
                dataset[:], abs_chern_nums=(chern_number,), verbose=verbose
            )
            n_avg, n_std = [], []
            u0s = sorted(list(set(u0_factors)))
            for u0 in u0s:
                ns = n_evals[u0_factors == u0]
                n_avg.append(np.mean(ns))
                n_std.append(np.std(ns))

            figure_data[f"chern_number_{chern_number}"][dataset_name] = {}
            figure_data[f"chern_number_{chern_number}"][dataset_name]["u0s"] = u0s
            figure_data[f"chern_number_{chern_number}"][dataset_name]["n_avg"] = n_avg
            figure_data[f"chern_number_{chern_number}"][dataset_name]["n_std"] = n_std

    with open(
        os.path.join(figure_path, "fig_num_states_disorder_data.json"), "w"
    ) as file:
        json.dump(figure_data, file)


def make_number_of_states_plot(figure_path: str):

    with open(
        os.path.join(figure_path, "fig_num_states_disorder_data.json"), "r"
    ) as file:
        figure_data = json.load(file)

    rc("text", usetex=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.9))
    colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:purple"]

    for chern_number in [0, 1, 2, 3]:
        for dataset_name, ax in zip(
            ["Training", "Test"],
            [ax1, ax2],
        ):
            u0s = np.array(
                figure_data[f"chern_number_{chern_number}"][dataset_name]["u0s"]
            )
            n_avg = np.array(
                figure_data[f"chern_number_{chern_number}"][dataset_name]["n_avg"]
            )
            n_std = np.array(
                figure_data[f"chern_number_{chern_number}"][dataset_name]["n_std"]
            )

            ax.plot(
                u0s,
                n_avg,
                "o",
                label=rf"$|C|={chern_number}$",
                color=colors[chern_number],
            )
            ax.fill_between(
                u0s,
                n_avg - n_std,
                n_avg + n_std,
                alpha=0.12 * (4 - chern_number),
                color=colors[chern_number],
            )

    ymax = 40
    for ax in [ax1, ax2]:
        ax.legend(loc="upper left", fontsize=14, ncol=1)
        ax.set_ylabel("number of states", fontsize=20)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 1)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=5)

    ax1.set_xlabel(r"$V_0/\bar \Delta_{\rm{training}}$", fontsize=20)
    ax2.set_xlabel(r"$V_0/\bar \Delta_{\rm{shiba}}$", fontsize=20)
    ax1.text(-0.19, 38.4, "a)", fontsize=24)
    ax2.text(-0.19, 38.4, "b)", fontsize=24)

    fig.subplots_adjust(wspace=0.3)

    for ftype in ["png", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "num_states_disorder." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
