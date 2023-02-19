# Copyright (c) 2020-2023, Paul Baireuther
# All rights reserved.

# Python standard library
from typing import List, Optional, Tuple, Union

# Third party
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


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


def fit_exp_decay(scale: float, x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Fits an exponentially decaying curve to data and returns mean squared error. Note: It is
    assumed, that `y[0] = f(0)` and `y[infinity] = 0`.

    Parameters
    ----------
    scale: Scale on which the exponential decay occurs
    x_values: x-values
    y_values: y-values

    Returns
    -------
    Mean square error
    """

    mse = np.mean((y_values - exp_decay(x_values, scale, y_values[0], 0)) ** 2)
    return mse


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


def estimate_localization_length(
    dataset: List,
    relative_gap_min: float = 0.0,
    relative_gap_max: float = np.inf,
    abs_chern_numbers: Tuple = (1, 2, 3),
    relative_disorder_strength_max: float = np.inf,
    title: str = "",
    filepath: Optional[str] = None,
    plotting: bool = False,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """Estimate the localization length using strips of LDOS away form the corners. For a 8x8
    grid shown below, the x denote sites that are considered and the o denote sites that are
    ignored.

    o  o  x  x  x  x  o  o
    o  o  x  x  x  x  o  o
    x  x  x  x  x  x  x  x
    x  x  x  x  x  x  x  x
    x  x  x  x  x  x  x  x
    x  x  x  x  x  x  x  x
    o  o  x  x  x  x  o  o
    o  o  x  x  x  x  o  o

    Parameters
    ----------
    dataset: A dataset
    relative_gap_min: Minimal gap to bandwidth ratio to be considered
    relative_gap_max: Maximal gap to bandwidth ratio to be considered
    abs_chern_nums: The absolute values of the Chern numbers to be considered
    relative_disorder_strength_max: Maximal disorder strength to bandwidth ratio to be considered
    title: Title shown on figure
    filepath: Filepath of figure
    plotting: If True, a figure will be generated
    verbose: If `True`, feedback will be printed


    Returns
    -------
    Localization length estimated by a fit from the boundary to the center,
    Localization length estimated by a fit with an offset from the boundary to the center,
    The offset used in the second fit.
    """

    sample_width_x = dataset[0]["system_params"]["num_sites_x"]
    sample_width_y = dataset[0]["system_params"]["num_sites_y"]
    if sample_width_x == sample_width_y:
        sample_width = sample_width_x
    else:
        raise NotImplementedError(
            "This function only works for square shaped LDOS maps."
        )
    if sample_width // 4 * 4 != sample_width:
        raise NotImplementedError(
            "This function only works for sample width that are a multiple of four."
        )

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
            "Given the current constraints, there are no valid samples in the "
            "dataset."
        )
    if verbose:
        print(f"No of samples is {num_selected_samples}.")
    ldos_array = ldos_array.reshape(-1, sample_width_x, sample_width_y)[condition]

    # A strip through the center can be done both horizontally and vertically. And for each
    # there are four possible orientations connected by mirror symmetries. For maximal
    # smoothness, average over all eight strips.
    strip_width = sample_width // 2  # Stripwidth is half the sample width
    strip_x = np.mean(ldos_array[:, :, strip_width // 2 : -strip_width // 2], axis=2)
    strip_y = np.mean(ldos_array[:, strip_width // 2 : -strip_width // 2, :], axis=1)
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

    # Fit an exponential decay to the averaged_strip
    # a) all the way from the boundary to the middle of the sample,
    half_strip = averaged_strip[: sample_width // 2]
    loc_length_from_boundary = minimize(
        fit_exp_decay,
        x0=1.0,
        args=(np.arange(half_strip.shape[0]), half_strip),
        tol=1e-15,
    ).x.item()
    # b) with an offset from the boundary.
    offset = sample_width // 8
    partial_strip = averaged_strip[offset : sample_width // 2]
    loc_length_away_from_boundary = minimize(
        fit_exp_decay,
        x0=1.0,
        args=(np.arange(partial_strip.shape[0]), partial_strip),
        tol=1e-15,
    ).x.item()

    if plotting:
        # Plot the LDOS in the average strip
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # In the first panel, some example strips are shown, and the average strip
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

        # In the second and third panel, the localization length fits are shown
        grid_from_boundary = np.linspace(0, len(half_strip) - 1, 101)
        vals_fit_from_boundary = exp_decay(
            x=grid_from_boundary,
            scale=loc_length_from_boundary,
            y0=half_strip[0],
            yinf=0,
        )
        vals_fit_with_offset_from_boundary = exp_decay(
            x=grid_from_boundary - offset,
            scale=loc_length_away_from_boundary,
            y0=partial_strip[0],
            yinf=0,
        )
        for n in (1, 2):
            ax = axes[n]
            ax.plot(
                range(1, half_strip.shape[0] + 1),
                half_strip,
                "o",
                label="averaged LDOS " "along strip",
            )
            ax.plot(
                grid_from_boundary + 1,
                vals_fit_from_boundary,
                color="tab:blue",
                label="loc length fitted from boundary to center is {}".format(
                    round(loc_length_from_boundary, 2)
                ),
            )
            ax.plot(
                grid_from_boundary + 1,
                vals_fit_with_offset_from_boundary,
                color="tab:red",
                label="loc length fitted with offset from boundary is {}".format(
                    round(loc_length_away_from_boundary, 2)
                ),
            )
            ax.set_xlim(1, sample_width // 2)
            ax.set_xlabel("position (1 is on the boundary)")
            ax.set_title("average LDOS with localization length fits")
            ax.legend()

        for n in range(3):
            axes[n].set_ylabel("averaged LDOS")

        axes[1].set_ylim(0, np.max(half_strip))
        axes[2].set_ylim(np.min(half_strip), np.max(half_strip))
        axes[2].set_yscale("log")

        fig.tight_layout(rect=[0.00, 0.04, 1.00, 0.98])
        if filepath:
            plt.savefig(filepath + ".png", facecolor="white")
        plt.close()

    if verbose:
        print(
            f"The localization length estimated by a fit from the boundary to the center is "
            f"{loc_length_from_boundary}."
        )
        print(
            f"The localization length estimated by a fit with a {offset} site offset from "
            f"boundary is {loc_length_away_from_boundary}."
        )

    return loc_length_from_boundary, loc_length_away_from_boundary, offset
