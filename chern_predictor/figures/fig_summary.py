# Copyright (c) 2020-2023, Paul Baireuther
# All rights reserved

# Python standard library
import os

# Third party libraries
from matplotlib import pyplot as plt
from matplotlib import rc

# This project
from chern_predictor.figures.fig_performance import make_performance_plots
from chern_predictor.figures.fig_threshold import make_threshold_figure


def make_summary_fig(
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
        figsize=(25, 4.95),
        gridspec_kw={"width_ratios": [3.2, 1.0, 1.3, 3.5, 1.4, 3]},
    )
    axes = axes.flatten()

    # Training curves and confusion matrix
    make_performance_plots(
        ax_training=axes[0],
        ax_histogram=axes[1],
        ax_confusion_matrix=axes[5],
        figure_path=figure_path,
        verbose=verbose,
    )

    # Certainty threshold
    twax = axes[3].twinx()
    twax.yaxis.tick_right()
    make_threshold_figure(
        ax_accuracy=axes[3], ax_fraction=twax, figure_path=figure_path
    )

    # Spacers
    axes[2].axis("off")
    axes[4].axis("off")

    # Misc
    fig.subplots_adjust(wspace=0)
    axes[0].text(-11.8, 1, r"a)", fontsize=fontsize)
    axes[3].text(-0.22, 1, r"b)", fontsize=fontsize)
    axes[5].text(-1.2, 3.5, r"c)", fontsize=fontsize)

    for ftype in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "summary." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
