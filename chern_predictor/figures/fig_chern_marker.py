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
import os

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def make_chern_marker_plot(figure_path: str):

    rc("text", usetex=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.9))
    colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:purple"]

    data_path = "chern_predictor/figures/data/"
    fname_c1 = "Chern_marker_scaling_vs_N_x_abs_C.1_kf_3.67900_e0_0.35400_gap0_0.30374_bandwidth_1.70100.txt"
    fname_c2 = "Chern_marker_scaling_vs_N_x_abs_C.2_kf_2.87600_e0_0.28800_gap0_0.13449_bandwidth_1.56362.txt"
    fname_c3 = "Chern_marker_scaling_vs_N_x_abs_C.3_kf_3.48700_e0_0.16200_gap0_0.15893_bandwidth_1.34522.txt"
    for n, fname in enumerate([fname_c1, fname_c2, fname_c3]):
        c = n + 1
        data = np.loadtxt(data_path + fname)
        ax1.plot(data[:, 0], data[:, 1], "o", label=rf"$|C|={c}$", color=colors[c])

    fname_c1 = "Chern_marker_scaling_vs_V_0_abs_C.1_gap_balanced_aver.0.2192_kf_3.67900_e0_0.35400_gap0_0.30374_bandwidth_1.70100.txt"
    fname_c2 = "Chern_marker_scaling_vs_V_0_abs_C.2_gap_balanced_aver.0.2192_kf_2.87600_e0_0.28800_gap0_0.13449_bandwidth_1.56362.txt"
    fname_c3 = "Chern_marker_scaling_vs_V_0_abs_C.3_gap_balanced_aver.0.2192_kf_3.48700_e0_0.16200_gap0_0.15893_bandwidth_1.34522.txt"

    # The data for the average clean gap used in calculating how the Chern marker
    # decays as a function of disorder was based on an older version of the dataset
    # which deviated by about 1% from the value in the final dataset. The following
    # scaling factor compensates for this, making this plot consistent with the other
    # figures.
    x_axis_scaling_factor = 0.2192 / 0.2217

    for n, fname in enumerate([fname_c1, fname_c2, fname_c3]):
        c = n + 1
        data = np.loadtxt(data_path + fname)
        ax2.plot(
            data[:, 0] * x_axis_scaling_factor,
            data[:, 1],
            "o",
            label=rf"$|C|={c}$",
            color=colors[c],
        )
        ax2.fill_between(
            data[:, 0] * x_axis_scaling_factor,
            data[:, 1] - data[:, 2],
            data[:, 1] + data[:, 2],
            alpha=0.12 * (4 - c),
            color=colors[c],
        )

    for ax in [ax1, ax2]:
        ax.set_ylabel(r"Chern marker $|C_m|$", fontsize=20)
        ax.set_ylim(0, 3)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.locator_params(axis="x", nbins=7)
        ax.locator_params(axis="y", nbins=4)

    ax1.set_xlabel(r"system size $N_x=N_y$", fontsize=20)
    ax1.set_xlim(10, 40)
    ax2.set_xlabel(r"disorder amplitude $V_0 / \bar \Delta_{\rm{shiba}}$", fontsize=20)
    ax2.set_xlim(0, 3)
    ax2.legend(loc="upper right", fontsize=14, ncol=1)
    ax1.text(+5.40, 2.93, "a)", fontsize=24)
    ax2.text(-0.45, 2.93, "b)", fontsize=24)

    fig.subplots_adjust(wspace=0.25)

    for ftype in ["png", "svg"]:
        fig.savefig(
            os.path.join(figure_path, "chern_marker." + ftype),
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close()
