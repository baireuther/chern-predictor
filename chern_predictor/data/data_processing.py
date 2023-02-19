# Copyright (c) 2020-2023, Paul Baireuther
# All rights reserved.

# Python standard library
from typing import List

# Third party
import numpy as np


def balance_dataset(
    data: List,
    chern_min: int,
    chern_max: int,
    gap_min: float,
    gap_is_relative: bool,
    forbidden_seeds: dict = {},
) -> list:
    """Generate a balanced dataset, in the sense that there will be an equal number of samples
    for each Chern number

    Parameters
    ----------
    data: List with the info about the Hamiltonians
    chern_min: The smallest Chern number to be considered
    chern_max: The largest Chern number to be considered
    gap_min: The minimal gap required
    gap_is_relative: If True, the relative gap = gap/bandwidth is used as gap
    forbidden_seeds: These seeds will be ignored

    Returns
    -------
    A list with the selected seeds
    """

    # Get a count of how many Hamiltonians exist for each Chern number
    counter = np.zeros(chern_max + 1, dtype=int)
    chern_numbers = {}
    for idx, datapoint in enumerate(data):
        seed = datapoint["ham_seed"]
        if seed not in forbidden_seeds:
            if gap_is_relative:
                gap = datapoint["relative_gap"]
            else:
                gap = datapoint["absolute_gap"]
            if gap >= gap_min:
                chern_no = int(np.rint(abs(datapoint["chern_number"])))
                if chern_no <= chern_max:
                    counter[chern_no] += 1
                    chern_numbers[idx] = chern_no

    # Print the results
    for n in range(counter.shape[0]):
        print("Chern {}: {}".format(n, counter[n]))

    # Figure out the maximal number of entries possible, while still being balanced
    n_max = int(np.min(counter[chern_min : chern_max + 1]))

    # Generate a balanced dataset
    counter2 = np.zeros(int(chern_max + 1), dtype=int)
    balanced_seeds = []
    for seed in chern_numbers:
        chern_no = chern_numbers[seed]
        if chern_no >= chern_min and chern_no <= chern_max:
            if counter2[chern_no] < n_max:
                balanced_seeds.append(seed)
                counter2[chern_no] += 1
    print(
        "Selected {} out of {} Hamiltonians".format(len(balanced_seeds), sum(counter)),
        end="\n\n",
    )

    return balanced_seeds
