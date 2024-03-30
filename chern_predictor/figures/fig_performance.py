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
import os.path

# Third party libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.stats import norm

# This project
from chern_predictor.networks.ensembles import evaluate_ensemble
from chern_predictor.networks.helper_functions import load_data, preprocess_dataset
from default_parameters import evaluation_params, network_parameters


def gen_performance_plot_data(
    ensemble_path: str,
    ensemble_seed: int,
    model_path: str,
    dataset_directory: str,
    figure_path: str,
):

    data = {}
    data["histogram"] = {}
    data["training"] = {}
    data["confusion_matrix"] = {}

    # Load the data
    _, _, data_test = load_data(
        dataset_directory,
        load_training_data=False,
        load_validation_data=False,
        load_test_data=True,
        verbose=False,
    )
    _, y_test, _ = preprocess_dataset(data_test, normalize=True)

    # Load precalculated predictions
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

    # The histogram
    ymin, ymax = 0.85, 1.0
    n_bins = 45
    bins = np.linspace(ymin, ymax, n_bins + 1)
    step_size = (ymax - ymin) / n_bins
    n_epochs = weights_test.shape[1] - 1
    data["n_epochs"] = n_epochs
    data["histogram"]["bins"] = bins.tolist()
    data["histogram"]["step_size"] = step_size
    num_bootstraps = evaluation_params["num_bootstraps"]
    min_num_epochs = evaluation_params["epochs_until_trained"] + 1

    for n, ensemble_size in enumerate(evaluation_params["ensemble_sizes"]):
        accs_mean, accs_std, accs, confusion_matrices = [], [], [], []
        for epoch in range(n_epochs + 1):
            acc, acc_mean, acc_std, _, _, _, cm_mean, _ = evaluate_ensemble(
                weights=weights_test[:, epoch, :, :],
                ground_truth=y_test,
                ensemble_size=ensemble_size,
                num_bootstraps=num_bootstraps,
                unique_ensembles=True,
                minimum_weight=-1,
                ensemble_seed=ensemble_seed + n,
            )
            accs.append(acc)
            accs_mean.append(acc_mean)
            accs_std.append(acc_std)
            confusion_matrices.append(cm_mean)
        accs = np.array(accs)[min_num_epochs:].flatten()
        accs_mean = np.array(accs_mean)
        accs_std = np.array(accs_std)
        confusion_matrix = np.mean(confusion_matrices[min_num_epochs:], axis=0)

        data["training"][f"ensemble_size_{ensemble_size}"] = {}
        data["confusion_matrix"][f"ensemble_size_{ensemble_size}"] = {}
        data["training"][f"ensemble_size_{ensemble_size}"]["accs"] = accs.tolist()
        data["training"][f"ensemble_size_{ensemble_size}"][
            "accs_mean"
        ] = accs_mean.tolist()
        data["training"][f"ensemble_size_{ensemble_size}"][
            "accs_std"
        ] = accs_std.tolist()
        data["confusion_matrix"][f"ensemble_size_{ensemble_size}"][
            "data"
        ] = confusion_matrix.tolist()

    with open(os.path.join(figure_path, "fig_performance_data.json"), "w") as file:
        json.dump(data, file)


def make_performance_plots(
    ax_training,
    ax_histogram,
    ax_confusion_matrix,
    figure_path: str,
    verbose: bool = False,
):

    with open(os.path.join(figure_path, "fig_performance_data.json"), "r") as file:
        figure_data = json.load(file)

    n_epochs = figure_data["n_epochs"]
    bins = figure_data["histogram"]["bins"]
    step_size = figure_data["histogram"]["step_size"]

    # Config
    fontsize = 28
    colors = evaluation_params["colors"]
    rc("text", usetex=True)

    # The histogram
    ymin, ymax = 0.85, 1.0

    overall_max_count = 0
    for n, ensemble_size in enumerate(evaluation_params["ensemble_sizes"]):
        accs = np.array(
            figure_data["training"][f"ensemble_size_{ensemble_size}"]["accs"]
        )
        accs_mean = np.array(
            figure_data["training"][f"ensemble_size_{ensemble_size}"]["accs_mean"]
        )
        accs_std = np.array(
            figure_data["training"][f"ensemble_size_{ensemble_size}"]["accs_std"]
        )
        confusion_matrix = np.array(
            figure_data["confusion_matrix"][f"ensemble_size_{ensemble_size}"]["data"]
        )

        label = f"n={ensemble_size}"
        if verbose:
            print(
                f"ensemble size: {ensemble_size}, accuracy: {round(np.mean(accs), 3)} +- {round(np.std(accs), 3)}"
            )
        ax_training.plot(range(n_epochs + 1), accs_mean, label=label, color=colors[n])
        ax_training.fill_between(
            range(n_epochs + 1),
            accs_mean - accs_std,
            accs_mean + accs_std,
            color=colors[n],
            alpha=0.2,
        )
        counts, _ = np.histogram(
            a=accs,
            bins=bins,
            weights=1.0 / (len(accs) * step_size) * np.ones(len(accs)),
        )
        overall_max_count = max(overall_max_count, np.max(counts))
        ax_histogram.hist(
            accs,
            bins=bins,
            color=colors[n],
            weights=1.0 / (len(accs) * step_size) * np.ones(len(accs)),
            alpha=0.4,
            orientation="horizontal",
        )
        xs = np.linspace(ymin, ymax, 201)
        ys = norm.pdf(xs, loc=np.mean(accs), scale=np.std(accs))
        ax_histogram.plot(xs, ys, "--", label=label, color=colors[n], lw=2)

    ax_training.hlines(
        [0.9, 0.95], 0, network_parameters["num_epochs"], linestyle="--", color="gray"
    )
    ax_training.set_xlim(0, network_parameters["num_epochs"])
    ax_training.set_ylim(0.85, 1)
    ax_histogram.set_xlim(0, 1.2 * overall_max_count)
    ax_histogram.set_ylim(0.85, 1)
    for ax in [ax_training, ax_histogram]:
        ax.xaxis.set_tick_params(labelsize=fontsize - 4)
        ax.yaxis.set_tick_params(labelsize=fontsize - 4)

    ax_training.set_ylabel("prediction accuracy", fontsize=fontsize)
    ax_training.set_xlabel("training epoch", fontsize=fontsize)
    ax_training.vlines(
        evaluation_params["epochs_until_trained"] + 0.5,
        0,
        1,
        linestyle="dashed",
        color="gray",
    )
    ax_training.locator_params(axis="y", nbins=4)
    ax_training.legend(ncol=2, loc="lower left", fontsize=fontsize - 4)

    ax_histogram.set_yticks([])
    ax_histogram.yaxis.set_label_position("right")
    ax_histogram.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    for line in ax_histogram.lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        line.set_xdata(ydata)
        line.set_ydata(xdata)

    cmap = "Blues"
    _ = ax_confusion_matrix.imshow(
        confusion_matrix, vmin=0, vmax=1, origin="lower", cmap=cmap
    )

    ax_confusion_matrix.locator_params(axis="both", nbins=4)
    ax_confusion_matrix.set_aspect("equal", "box")
    ax_confusion_matrix.set_xlabel("true $|C|$", fontsize=fontsize)
    ax_confusion_matrix.set_ylabel("predicted $|C|$", fontsize=fontsize)
    for tl in ax_confusion_matrix.get_xticklabels():
        tl.set_color("k")
        tl.set_size(fontsize - 4)
    for tl in ax_confusion_matrix.get_yticklabels():
        tl.set_color("k")
        tl.set_size(fontsize - 4)

    for i in range(4):
        for j in range(4):
            if i == j:
                ax_confusion_matrix.text(
                    i,
                    j,
                    round(100 * confusion_matrix[j, i], 1),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=fontsize - 6,
                )
            else:
                ax_confusion_matrix.text(
                    i,
                    j,
                    round(100 * confusion_matrix[j, i], 1),
                    ha="center",
                    va="center",
                    color="k",
                    fontsize=fontsize - 6,
                )


def make_performance_fig(
    figure_path: str,
    verbose: bool = False,
):
    # Config
    fontsize = 28
    rc("text", usetex=True)

    # Make figure
    fig, axes = plt.subplots(
        1,
        6,
        figsize=(24.65, 4.93),
        gridspec_kw={"width_ratios": [3.5, 1, 3.2, 1, 0.75, 3]},
    )
    axes = axes.flatten()
    axes[0].axis("off")

    make_performance_plots(
        ax_training=axes[2],
        ax_histogram=axes[3],
        ax_confusion_matrix=axes[5],
        figure_path=figure_path,
        verbose=verbose,
    )

    # Misc
    axes[1].axis("off")
    axes[4].axis("off")

    # axes[0].text(-35, 1, r"a)", fontsize=fontsize)
    axes[2].text(-13, 1, r"b)", fontsize=fontsize)
    axes[5].text(-1.2, 3.5, r"c)", fontsize=fontsize)

    fig.subplots_adjust(wspace=0)

    for ftype in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "performance." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
