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
from typing import List, Optional, Tuple, Union

# Third party
import numpy as np
from matplotlib import pyplot as plt


def exp_decay(
    x: Union[float, np.ndarray], scale: float, y0: float, yinf: float
) -> Union[float, np.ndarray]:
    """Exponentially decaying curve.

    Parameters
    ----------
    x: Input values
    scale: Scale on which the exponential decay occurs
    y0: Function value at `x=0`
    yinf: Function value at `x=infinity`

    Returns
    -------
    Function value(s)
    """
    return (y0 - yinf) * np.exp(-x / scale) + yinf


def extract_data_from_dataset(
    dataset: List, abs_chern_nums: Tuple = (0, 1, 2, 3), verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to extract info from a dataset.

    Parameters
    ----------
    dataset: A dataset
    abs_chern_nums: The absolute values of the Chern numbers that will be considered
    verbose: If `True`, feedback will be printed

    Returns
    -------
    number of eigenvalues in LDOS window, bandgap to bandwidth ratio, normalized LDOS,
    relative disorder strengths
    """

    (
        num_states_in_ldos_window_list,
        relative_gap_list,
        normalized_ldos_list,
        relative_disorder_strength_list,
    ) = ([], [], [], [])

    # Keep track of how often there are no states in the LDOS window
    num_datapoints_with_zero_ldos = 0
    for datapoint in dataset:
        abs_val_of_chern_number = datapoint["chern_number_absolute_value"]
        if abs_val_of_chern_number in abs_chern_nums:
            num_states_in_ldos_window = datapoint["num_states_in_ldos"]
            num_states_in_ldos_window_list.append(num_states_in_ldos_window)
            relative_gap_list.append(datapoint["relative_gap"])
            relative_disorder_strength_list.append(
                datapoint["relative_disorder_strength"]
            )
            ldos = datapoint["local_density_of_states"]

            if num_states_in_ldos_window > 0:
                # To be removed
                if np.min(ldos) <= 0:
                    raise ValueError(
                        f"The LDOS must be larger or equal to zero, "
                        f"but the LDOS for seed {datapoint['seed']} the minimal LDOS "
                        f"is {np.min(ldos)}."
                    )
                ldos_normalized = ldos / np.sum(ldos)
                normalized_ldos_list.append(ldos_normalized)
            else:
                assert len(ldos) > 0
                assert np.sum(ldos) == 0
                normalized_ldos_list.append(ldos)
                num_datapoints_with_zero_ldos += 1

    if verbose:
        num_datapoints = len(dataset)
        ratio_in_percent = round(
            100 * num_datapoints_with_zero_ldos / num_datapoints,
            2,
        )
        print(
            f"The number of cases with zero LDOS is {num_datapoints_with_zero_ldos} out of "
            f"{num_datapoints} which is {ratio_in_percent}%."
        )

    return (
        np.array(num_states_in_ldos_window_list),
        np.array(relative_gap_list),
        np.array(normalized_ldos_list),
        np.array(relative_disorder_strength_list),
    )


def calc_localization_lengths(
    depths: np.ndarray, average_ldos: np.ndarray
) -> np.ndarray:
    """Function to calculate estimates for the average localization lengths of the edge
    states by the  following method: Consider the ratio between the average LDOS at a
    certain distance from the boundary and the average LDOS at the smallest considered
    distance from the boundary. Then take the relative depth rel_depth and calculate an
    estimate of the localization length by loc_length = - rel_depth / log(r).

    Parameters
    ----------
    depths: The locations of the considered average LDOS relative to the boundary.
    average_ldos: The average value of the LDOS at a given corresponding depth.

    Returns
    -------
        localization length estimates
    """
    localization_length_estimates = []
    n_min_depth = np.argmin(depths)
    for depth, avg_ldos in zip(depths, average_ldos):
        rel_depth = depth - np.min(depths)
        if rel_depth > 0:
            ldos_ratio = avg_ldos / average_ldos[n_min_depth]
            localization_length_estimates.append(-rel_depth / np.log(ldos_ratio))
    return np.array(localization_length_estimates)


def estimate_localization_length(
    dataset: List,
    strip_width: int = 4,
    offset: int = 3,
    relative_gap_min: float = 0.0,
    relative_gap_max: float = np.inf,
    abs_chern_numbers: Tuple = (1, 2, 3),
    relative_disorder_strength_max: float = np.inf,
    filepath: Optional[str] = None,
    plotting: bool = False,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """Estimate the localization length using strips of LDOS away form the corners. For a
    12x12 grid shown below, v denote sites that are considered, x denote sites that are
    not considered because they are too close to the corners or other boundaries and o
    denote offset-sites that are also not considered, because they are very close to the
    boundary.

    x  x  x  x  x  o  o  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    o  v  v  v  v  x  x  v  v  v  v  o
    o  v  v  v  v  x  x  v  v  v  v  o
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  v  v  x  x  x  x  x
    x  x  x  x  x  o  o  x  x  x  x  x

    The resulting average strip is of shape v v v v and is used to calculate estimates
    of the localization length by considering all pairings between a site in the strip
    with the outermost site in the strip (i.e. the one closest to the boundary). Then
    the mean and the standard deviation are returned.


    Parameters
    ----------
    dataset: A dataset
    strip_width: Width of the strip to be considered
    offset: Additional offset from boundary
    relative_gap_min: Minimal gap to bandwidth ratio to be considered
    relative_gap_max: Maximal gap to bandwidth ratio to be considered
    abs_chern_nums: The absolute values of the Chern numbers to be considered
    relative_disorder_strength_max: Maximal disorder strength to bandwidth ratio to be considered
    filepath: Filepath of figure
    plotting: If True, a figure will be generated
    verbose: If `True`, feedback will be printed


    Returns
    -------
    Mean of localization length estimates
    Standard deviation of localization length estimates
    """

    sample_width_x = dataset[0]["system_params"]["num_sites_x"]
    sample_width_y = dataset[0]["system_params"]["num_sites_y"]
    if sample_width_x == sample_width_y:
        sample_width = sample_width_x
    else:
        raise NotImplementedError(
            "This function only works for square shaped LDOS maps."
        )
    if np.mod(strip_width, 2) != 0:
        raise NotImplementedError("The strip width must me a multiple of two.")
    if np.mod(sample_width, 2) != 0:
        raise NotImplementedError("The sample width must me a multiple of two.")

    # Extract LDOS maps, gap size, and disorder strength form the dataset for datapoints
    # satisfying the constraints.
    (
        _,
        relative_gaps,
        ldos_array,
        relative_disorder_strength,
    ) = extract_data_from_dataset(dataset, abs_chern_numbers, verbose=verbose)
    gap_condition_lower = relative_gaps >= relative_gap_min
    gap_condition_upper = relative_gaps <= relative_gap_max
    disorder_condition = relative_disorder_strength <= relative_disorder_strength_max
    condition = gap_condition_lower * gap_condition_upper * disorder_condition
    num_selected_samples = np.sum(condition)
    if num_selected_samples == 0:
        raise ValueError(
            "Given the current constraints, there are no valid samples in the dataset."
        )
    if verbose:
        print(f"No of considered samples is {num_selected_samples}.")
    ldos_array = ldos_array.reshape(-1, sample_width_x, sample_width_y)[condition]

    # A strip through the center can be done both horizontally and vertically. And for
    # each there are four possible orientations connected by mirror symmetries. For
    # maximal smoothness, average over all eight strips.
    sample_width_half = sample_width // 2
    strip_width_half = strip_width // 2
    strip_x = np.mean(
        ldos_array[
            :,
            :,
            sample_width_half - strip_width_half : sample_width_half + strip_width_half,
        ],
        axis=2,
    )
    strip_y = np.mean(
        ldos_array[
            :,
            sample_width_half - strip_width_half : sample_width_half + strip_width_half,
            :,
        ],
        axis=1,
    )
    strip_x_reversed = strip_x[:, ::-1]
    strip_y_reversed = strip_y[:, ::-1]
    all_strips = np.concatenate(
        [
            strip_x,
            strip_x_reversed,
            strip_y,
            strip_y_reversed,
        ],
        axis=0,
    )
    averaged_strip = np.mean(all_strips, axis=0)

    half_strip = averaged_strip[: sample_width_half - strip_width_half]
    partial_strip = averaged_strip[offset : sample_width_half - strip_width_half]
    loc_lengths = calc_localization_lengths(
        depths=np.arange(partial_strip.shape[0]), average_ldos=partial_strip
    )
    loc_length_mean = np.mean(loc_lengths)
    loc_length_std = np.std(loc_lengths)
    loc_length_min = np.min(loc_lengths)
    loc_length_max = np.max(loc_lengths)
    if verbose:
        print(
            f"The localization lengths, fit to the averaged half-strip with an offset "
            f"of {offset} from boundary and of strip-width / 2 from the center are "
            f"{np.around(loc_lengths, 2)} with "
            f"mean {round(loc_length_mean, 2)} and "
            f"standard deviation {round(loc_length_std, 2)}."
        )

    if plotting:
        # Plot the LDOS of the averaged half-strip from the boundary to the center minus
        # strip-width half sites.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Localization length estimate with offset {offset} from the boundary "
            f"(mean, standard deviation)=({round(loc_length_mean, 2)},"
            f"{round(loc_length_std, 2)}).",
            fontsize=16,
        )

        # In the first panel, some example strips are shown, and the average
        num_strips_in_plot = min(50, num_selected_samples)
        axes[0].plot(
            range(1, sample_width + 1),
            all_strips[:num_strips_in_plot].transpose(),
            color="grey",
        )
        axes[0].plot(range(1, averaged_strip.shape[0] + 1), averaged_strip, "k", lw=5)
        axes[0].set_xlim(1, sample_width)
        axes[0].set_ylim(0, 4 / (sample_width**2))
        axes[0].set_xlabel("position")
        axes[0].set_title("a couple of example strips and the averaged strip")

        # In the second and third panel, the localization length estimates are shown
        depths = np.linspace(0, sample_width_half - strip_width_half, 101)
        exp_curve_slowest_decay = exp_decay(
            x=depths - offset,
            scale=loc_length_max,
            y0=partial_strip[0],
            yinf=0,
        )
        exp_curve_fastest_decay = exp_decay(
            x=depths - offset,
            scale=loc_length_min,
            y0=partial_strip[0],
            yinf=0,
        )
        for n in (1, 2):
            ax = axes[n]
            ax.plot(
                range(1, half_strip.shape[0] + 1),
                half_strip,
                "o",
                color="tab:blue",
                label="averaged LDOS along half-strip",
            )
            ax.plot(
                range(offset + 1, half_strip.shape[0] + 1),
                half_strip[offset:],
                ".",
                color="tab:red",
                label="points considered in fits",
            )
            ax.plot(
                depths + 1,
                exp_curve_fastest_decay,
                "--",
                color="tab:red",
                label="shortest loc length {}".format(round(loc_length_min, 2)),
            )
            ax.plot(
                depths + 1,
                exp_curve_slowest_decay,
                "--",
                color="tab:red",
                label="longest loc length {}".format(round(loc_length_max, 2)),
            )
            ax.set_xlim(1, sample_width // 2)
            ax.set_xlabel("position (1 is on the boundary)")
            ax.set_title("average LDOS with localization length fits")
            ax.legend(loc="upper right")

        for n in range(3):
            axes[n].set_ylabel("averaged LDOS")

        axes[1].set_ylim(0, np.max(half_strip))
        axes[2].set_ylim(np.min(half_strip), np.max(half_strip))
        axes[2].set_yscale("log")

        fig.tight_layout(rect=[0.00, 0.04, 1.00, 0.98])
        if filepath:
            plt.savefig(filepath + f"_offset_{offset}.png", facecolor="white")
        plt.close()

    return loc_length_mean, loc_length_std
