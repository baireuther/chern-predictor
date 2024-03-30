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
from scipy.stats import norm

# This project
from chern_predictor.networks.ensembles import evaluate_ensemble
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset
from default_parameters import evaluation_params, network_parameters


def gen_running_average_plot_data(
    ensemble_path: str, model_path: str, dataset_directory: str, figure_path: str
):

    figure_data = {}

    _, _, data_test = load_data(dataset_directory, verbose=False)
    _, y_test, _ = preprocess_dataset(data_test, normalize=True)

    with open(ensemble_path + "network_list.json", "r") as file:
        model_name_list = json.load(file)

    weights_test = []
    for model_name in model_name_list:
        with open(
            os.path.join(model_path, model_name + "-evaluated.json"), "r"
        ) as file:
            accs_dict = json.load(file)
        w = accs_dict["test"]["prediction_vector"]
        weights_test.append(w)
    weights_test = np.array(weights_test)

    # SETTINGS
    n_nets = len(weights_test)
    n_epochs = weights_test.shape[1] - 1
    figure_data["n_epochs"] = n_epochs

    # Single net
    accs = []
    for epoch in range(n_epochs + 1):
        accs.append([])
        for n in range(n_nets):
            acc, _, _, _, _, _, _, _ = evaluate_ensemble(
                weights=weights_test[n : n + 1, epoch, :, :],
                ground_truth=y_test,
                ensemble_size=1,
                num_bootstraps=1,
                minimum_weight=-1,
            )
            accs[-1] += acc

    figure_data["1_average"] = {}
    figure_data["1_average"]["accs"] = accs

    # n-average
    for i, n_avg in enumerate([2, 4, 8]):
        accs = []
        for epoch in range(n_epochs + 1 - (n_avg - 1)):
            accs.append([])
            for n in range(n_nets):
                acc, _, _, _, _, _, _, _ = evaluate_ensemble(
                    weights=weights_test[
                        n : n + 1, epoch : epoch + n_avg, :, :
                    ].reshape(n_avg * 1, weights_test.shape[2], weights_test.shape[3]),
                    ground_truth=y_test,
                    ensemble_size=n_avg,
                    num_bootstraps=1,
                    minimum_weight=-1,
                )
                accs[-1] += acc

        figure_data[f"{n_avg}_average"] = {}
        figure_data[f"{n_avg}_average"]["accs"] = accs

    with open(os.path.join(figure_path, "fig_running_average_data.json"), "w") as file:
        json.dump(figure_data, file)


def make_running_average_fig(figure_path: str, verbose: bool = False):

    with open(os.path.join(figure_path, "fig_running_average_data.json"), "r") as file:
        figure_data = json.load(file)

    def plot_entry(n_epochs, accs, n_avg, ax1, ax2, color, name, ymin, ymax, n_bins=45):

        step_size = (ymax - ymin) / n_bins
        bins = np.linspace(ymin, ymax, n_bins + 1)

        accs = np.array(accs)
        accs_mean = np.mean(accs, axis=1)
        accs_std = np.std(accs, axis=1)
        accs = np.array(accs)[
            evaluation_params["epochs_until_trained"] + 1 - (n_avg - 1) :
        ].flatten()

        label = f"{name}"
        if verbose:
            print(label + f": {round(np.mean(accs), 3)} +- {round(np.std(accs), 3)}")

        # def plot_hist_entry(n_epochs, accs_mean, accs_std, ax1, ax2):
        n0 = n_epochs - len(accs_mean)
        ax1.plot(range(n0 + 1, n_epochs + 1), accs_mean, label=label, color=color)
        ax1.fill_between(
            range(n0 + 1, n_epochs + 1),
            accs_mean - accs_std,
            accs_mean + accs_std,
            color=color,
            alpha=0.2,
        )
        counts, _ = np.histogram(
            a=accs,
            bins=bins,
            weights=1.0 / (len(accs) * step_size) * np.ones(len(accs)),
        )
        ax2.hist(
            accs,
            bins=bins,
            color=color,
            weights=1.0 / (len(accs) * step_size) * np.ones(len(accs)),
            alpha=0.4,
            orientation="horizontal",
        )
        xs = np.linspace(ymin, ymax, 201)
        ys = norm.pdf(xs, loc=np.mean(accs), scale=np.std(accs))
        ax2.plot(xs, ys, "--", label=label, color=color, lw=2)

        return np.max(counts)

    ymin, ymax = 0.85, 1.0
    overall_max_count = 0

    rc("text", usetex=True)

    # SETTINGS
    colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.95), gridspec_kw={"width_ratios": [3.0, 1.2]}
    )
    axes = axes.flatten()

    # n-average
    for i, n_avg in enumerate([1, 2, 4, 8]):
        accs = figure_data[f"{n_avg}_average"]["accs"]
        max_count = plot_entry(
            n_epochs=figure_data["n_epochs"],
            accs=accs,
            n_avg=n_avg,
            ax1=axes[0],
            ax2=axes[1],
            color=colors[i],
            name=f"{n_avg}-avg",
            ymin=ymin,
            ymax=ymax,
        )
        overall_max_count = max(max_count, overall_max_count)

    # Further figure details
    for ax in axes[[0]]:
        ax.hlines(
            [0.9, 0.95],
            0,
            network_parameters["num_epochs"],
            linestyle="--",
            color="gray",
        )
        ax.set_xlim(0, network_parameters["num_epochs"])
        ax.set_ylim(ymin, ymax)
    for ax in axes[[1]]:
        # ax.set_xlim(0, 1.2 * overall_max_count)
        ax.set_xlim(0, 130)
        ax.set_ylim(ymin, ymax)
    for ax in axes[[0, 1]]:
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
    axes[0].locator_params(axis="x", nbins=6)

    for ax in axes[[1]]:
        ax.set_yticks([])

    axes[0].set_ylabel("prediction accuracy", fontsize=24)
    axes[1].yaxis.set_label_position("right")

    for ax in axes[[0]]:
        ax.set_xlabel("training epoch", fontsize=24)
        ax.vlines(
            [evaluation_params["epochs_until_trained"] + 0.5],
            0,
            1,
            linestyle="dashed",
            color="gray",
        )
        ax.locator_params(axis="y", nbins=4)

    axes[0].legend(ncol=2, loc="lower left", fontsize=16)

    axes[1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    for line in axes[1].lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        line.set_xdata(ydata)
        line.set_ydata(xdata)

    fig.subplots_adjust(wspace=0)

    for ftype in ["png", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "running_average." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
