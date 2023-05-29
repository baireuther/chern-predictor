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
import copy
import json
import multiprocessing as mp
import os
import time
from itertools import count
from typing import List, Optional, Tuple

# Third party libraries
import numpy as np
from scipy import sparse as sparse

# This project
from chern_predictor.data.chern_numbers import calc_chern_number
from chern_predictor.data.hamiltonian_coefficients import generate_ham_coefficients
from chern_predictor.data.ldos import calc_ldos_sparse, make_square
from chern_predictor.data.spectrum import calc_gap


def gen_dataset_name_and_experiment_directory(
    params: dict, storage_directory: str, suffix: str = ""
):
    dataset_name = "hopscale_{}_hopcutoff_{}_max_rel_pairing_{}_mingap_{}".format(
        str(params["bulk_ham"]["hopping_decay_scale"]).replace(".", "p"),
        str(params["bulk_ham"]["hopping_cutoff"]),
        str(params["bulk_ham"]["max_relative_pairing_strength"]).replace(".", "p"),
        str(np.round(params["bulk_ham"]["min_absolute_gap"], 3)).replace(".", "p")[:5],
    )
    if len(suffix) > 0:
        dataset_name = dataset_name + "_" + suffix
    experiment_directory = os.path.join(storage_directory, dataset_name)
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)
    return dataset_name, experiment_directory


def gen_bulk_hamiltonians(
    params: dict, dataset_path: str, parallelize: bool, verbose: bool
):

    # Start a little "cluster"
    if parallelize and params["data_generation"]["chunk_size"] > 1:
        pool = mp.Pool(mp.cpu_count())

    # Generate Hamiltonians in chunks of size `params["data_generation"]["chunk_size"]`.
    dataset = []
    smallest_ham_seed = params["dataset"]["smallest_ham_seed"]
    num_hams_per_chern_val = {}
    for c in params["dataset"]["chern_abs_vals"]:
        num_hams_per_chern_val[c] = 0

    # Continue the Hamiltonian generation cycle until the desired number of Hamiltonians per
    # value of the chern number are found.
    start_time = time.time()
    for iteration in count(start=1):

        # Define a list of tasks to be processed in this iteration
        arguments = []
        for ham_seed in range(
            smallest_ham_seed,
            smallest_ham_seed + params["data_generation"]["chunk_size"],
        ):
            arguments.append(
                [
                    params["bulk_ham"]["hopping_decay_scale"],
                    params["bulk_ham"]["hopping_cutoff"],
                    params["bulk_ham"]["max_relative_pairing_strength"],
                    params["bulk_ham"]["min_absolute_gap"],
                    ham_seed,
                ]
            )
        smallest_ham_seed += params["data_generation"]["chunk_size"]

        # Evaluate the tasks
        if parallelize and params["data_generation"]["chunk_size"] > 1:
            results = pool.starmap(evaluate_hamiltonian, arguments)
        else:
            results = [evaluate_hamiltonian(*arguments[0])]

        # Select the results that fulfill the conditions and store them
        for result in results:
            (
                relative_gap,
                absolute_gap,
                chern_number,
                relative_pairing_strength,
                ham_seed,
            ) = result

            # Check if conditions are fulfilled
            if absolute_gap >= params["bulk_ham"]["min_absolute_gap"]:
                if chern_number is None:
                    raise ValueError("Chern number was None.")
                else:
                    chern_number_absolut_value = int(abs(chern_number))
                if chern_number_absolut_value not in num_hams_per_chern_val:
                    continue
                if (
                    num_hams_per_chern_val[chern_number_absolut_value]
                    == params["dataset"]["num_hams_per_chern_number"]
                ):
                    continue
                else:
                    num_hams_per_chern_val[chern_number_absolut_value] += 1

                # Generate a datapoint
                datapoint = {
                    "bulk_ham_params": params["bulk_ham"],
                    "system_params": params["system"],
                    "dataset_params": params["dataset"],
                    "data_generation_params": params["data_generation"],
                    "ham_seed": int(ham_seed),
                    "relative_pairing_strength": relative_pairing_strength,
                    "absolute_gap": absolute_gap,
                    "relative_gap": relative_gap,
                    "bandwidth": absolute_gap / relative_gap,
                    "chern_number": chern_number,
                    "chern_number_absolute_value": chern_number_absolut_value,
                }
                dataset.append(datapoint)

        if verbose:
            # Print some feedback and save the generated data every now and then.
            print(
                "{} out of {} hamiltonians selected.".format(
                    len(dataset), iteration * params["data_generation"]["chunk_size"]
                )
            )
            print(num_hams_per_chern_val)
            elapsed_time = time.time() - start_time
            progress = (
                min(num_hams_per_chern_val.values())
                / params["dataset"]["num_hams_per_chern_number"]
            )
            print_progress_status(elapsed_time, progress)
            print()

        # Break the iteration cycle if the dataset is complete
        if (
            np.min(list(num_hams_per_chern_val.values()))
            == params["dataset"]["num_hams_per_chern_number"]
        ):
            print("Calculating of the Hamiltonian coefficients is complete.")
            break

    # Dump the dataset to file
    if type(dataset_path) == str:
        with open(dataset_path, "w") as file:
            json.dump(dataset, file)

    # Shut down "cluster"
    if parallelize and params["data_generation"]["chunk_size"] > 1:
        pool.close()

    return dataset


def generate_realizations(
    dataset_path: str, dataset: Optional[List] = None, verbose: bool = False
):

    # Load dataset
    if type(dataset_path) == str and os.path.exists(dataset_path):
        with open(dataset_path, "r") as file:
            dataset_from_file = json.load(file)
        if dataset is not None:
            assert len(dataset) == len(dataset_from_file)
            for n in range(len(dataset)):
                assert dataset[n] == dataset_from_file[n]
        else:
            dataset = dataset_from_file

    # Calculate the average bulk gap (needed to scale the disorder) and update the dataset
    gaps, bandwidths = [], []
    for data_point in dataset:
        gaps.append(data_point["absolute_gap"])
        bandwidths.append(data_point["bandwidth"])
    average_gap = np.mean(gaps)
    average_bandwidth = np.mean(bandwidths)
    for data_point in dataset:
        data_point["dataset_params"]["average_gap"] = average_gap
        data_point["dataset_params"]["average_bandwidth"] = average_bandwidth
    if verbose:
        print(f"Average bandwidth is {round(average_bandwidth, 3)}.")
        print(f"Average gap is {round(average_gap, 3)}.\n")

    assert (
        dataset[0]["system_params"]["num_sites_x"]
        == dataset[0]["system_params"]["num_sites_y"]
    )

    # Generate and evaluate the disorder realizations
    t0 = time.time()
    dataset_with_ldos = []
    for n, dat in enumerate(dataset):

        xi, delta, _ = generate_ham_coefficients(
            hopping_scale=dat["bulk_ham_params"]["hopping_decay_scale"],
            hopping_cutoff=dat["bulk_ham_params"]["hopping_cutoff"],
            max_relative_pairing_strength=dat["bulk_ham_params"][
                "max_relative_pairing_strength"
            ],
            rng=np.random.RandomState(int(dat["ham_seed"])),
        )

        # Every now and then, check if the correct Hamiltonians are used
        if n % 100 == 0:
            relative_gap, absolute_gap = calc_gap(
                xi=xi,
                delta=delta,
                seed=dat["ham_seed"],
                min_absolute_gap=dat["bulk_ham_params"]["min_absolute_gap"],
            )
            assert relative_gap == dat["relative_gap"]
            assert absolute_gap == dat["absolute_gap"]

        edge_length = dat["system_params"]["num_sites_x"]
        hopping_cutoff = dat["bulk_ham_params"]["hopping_cutoff"]
        ham0 = make_square(
            edge_length=edge_length,
            hopping_cutoff=hopping_cutoff,
            xi=xi,
            delta=delta,
        )

        rng = np.random.RandomState(seed=dat["ham_seed"])
        for relative_disorder_strength in dat["dataset_params"][
            "relative_disorder_strengths"
        ]:
            disorder_seed = rng.randint(0, 10**9)
            rng_local = np.random.RandomState(disorder_seed)

            # Generate disorder configuration
            disorder_strength = relative_disorder_strength * average_gap
            disorder = disorder_strength * (
                1 - 2 * rng_local.rand(edge_length * edge_length)
            )
            disorder = disorder.reshape(-1, 1) * np.array([1, -1]).reshape(1, 2)
            disorder = disorder.flatten()
            disorder = sparse.diags(disorder)
            ham = ham0 + disorder

            # Calculate eigenvalues and LDOS
            eigenvalues_in_ldos, ldos = calc_ldos_sparse(
                ham=ham,
                en_lim=dat["dataset_params"]["ldos_window_as_fraction_of_gap"]
                * (average_gap / 2.0),
                en_resolved=False,
                kmin=24,
                seed=disorder_seed,
                verbose=False,
            )

            # Store the data
            dat_with_ldos = copy.deepcopy(dat)
            dat_with_ldos["disorder_seed"] = disorder_seed
            dat_with_ldos["relative_disorder_strength"] = relative_disorder_strength
            dat_with_ldos["eigenvalues_in_ldos"] = eigenvalues_in_ldos.tolist()
            dat_with_ldos["num_states_in_ldos"] = len(
                dat_with_ldos["eigenvalues_in_ldos"]
            )
            dat_with_ldos["local_density_of_states"] = ldos.tolist()
            dataset_with_ldos.append(dat_with_ldos)

        # Print some feedback about the progress
        if verbose:
            if n % 1000 == 999 or n + 1 == len(dataset):
                print(f"Processed {n + 1} of {len(dataset)} Hamiltonians")
                elapsed_time = time.time() - t0
                progress = (n + 1) / len(dataset)
                print_progress_status(elapsed_time, progress)
                print()
    if verbose:
        print("Calculating the system realizations is complete.")

    # Dump the dataset to file
    if type(dataset_path) == str:
        with open(dataset_path, "w") as file:
            json.dump(dataset_with_ldos, file)

    return dataset_with_ldos


def evaluate_hamiltonian(
    hopping_scale: float,
    hopping_cutoff: float,
    max_relative_pairing_strength: float,
    min_absolute_gap: float,
    seed: int,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """This function generates and evaluates a bulk Hamiltonian.

    Parameters
    ----------
    hopping_scale: The length scale over which the hopping amplitude decays as
                   ~exp(-distance/hopping_lengthscale)
    hopping_cutoff: The maximal hopping distance, i.e. dx^2 + dy^2 <= hopping_cutoff.
    max_relative_pairing_strength: The maximum allowed ratio of pairing amplitudes to normal hoppings
    min_absolute_gap: The minimal required gap in absolute units; For Hamiltonians with gap below
                      the minimal gap, the calculation is not completed.
    seed: The seed
    verbose: If true, some feedback is printed

    Returns
    -------
    gap_relative: The size of the gap relative to the bandwidth
    gap_absolute: The size of the gap in absolute units
    chern_no: The Chern number
    seed: The seed that was used
    """

    # First, generate the hoppings and pairings
    Xi, Delta, relative_pairing_strength = generate_ham_coefficients(
        hopping_scale=hopping_scale,
        hopping_cutoff=hopping_cutoff,
        max_relative_pairing_strength=max_relative_pairing_strength,
        rng=np.random.RandomState(seed),
        verbose=verbose,
    )

    # Calculate the gap
    relative_gap, absolute_gap = calc_gap(
        xi=Xi,
        delta=Delta,
        seed=seed,
        min_absolute_gap=min_absolute_gap,
        verbose=verbose,
    )

    # If the gap is large enough, calculate the Chern number
    chern_number = None
    if absolute_gap >= min_absolute_gap:
        chern_number = calc_chern_number(
            Xi=Xi,
            Delta=Delta,
            conv_margin=1e-2,
            n_quad_min=100,
            n_quad_max=1600,
            verbose=verbose,
        )

    return relative_gap, absolute_gap, chern_number, relative_pairing_strength, seed


def print_progress_status(elapsed_time: float, progress: float):
    if progress > 0:
        elapsed_hrs = int(elapsed_time // 3600)
        elapsed_minutes = int((elapsed_time - elapsed_hrs * 3600) // 60)
        elapsed_seconds = int(elapsed_time - elapsed_hrs * 3600 - elapsed_minutes * 60)

        estimated_remaining_time = (1 - progress) * elapsed_time / progress
        remaining_hrs = int(estimated_remaining_time // 3600)
        remaining_minutes = int((estimated_remaining_time - remaining_hrs * 3600) // 60)
        remaining_seconds = int(
            estimated_remaining_time - remaining_hrs * 3600 - remaining_minutes * 60
        )

        print(
            "Elapsed time {}:{}:{}, estimated time to completion is {}:{}:{}.".format(
                elapsed_hrs,
                elapsed_minutes,
                elapsed_seconds,
                remaining_hrs,
                remaining_minutes,
                remaining_seconds,
            )
        )
