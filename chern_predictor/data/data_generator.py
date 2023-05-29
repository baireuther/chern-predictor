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
from typing import List, Optional

# This project
from chern_predictor.data.data_generator_functions import (
    gen_bulk_hamiltonians,
    generate_realizations,
)


def gen_dataset(
    params: dict,
    dataset_path: Optional[str] = None,
    parallelize: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """Function to generate a dataset in an reproducible way so that the dataset is fully
    specified by the params. Note that there can be small discrepancies in the generated data
    depending on e.g. the operating system.

    Parameters
    ----------
    params: The parameters specifying the dataset uniquely.

        params = {
            "bulk_ham": bulk_ham_params,
            "system": system_params,
            "dataset": dataset_params,
            "data_generation": data_generation_params,
        }

        # Bulk Hamiltonian parameters
        bulk_ham_params = {
            "hopping_cutoff": int,
            "hopping_decay_scale": float,
            "max_relative_pairing_strength": float,
            "min_absolute_gap": float,
        }

        # System parameters
        system_params = {
            "num_sites_x": int,
            "num_sites_y": int,
            "geometry": "rectangle with hard wall boundary condition",
        }

        # Dataset parameters
        dataset_params = {
            "chern_abs_vals": List[int],
            "ldos_window_as_fraction_of_gap": float,
            "relative_disorder_strengths": List[float],
            "smallest_ham_seed": int,
            "num_hams_per_chern_number": int,
            "git_hash": str,
        }

        # Dataset generation parameters
        data_generation_params = {"chunk_size": int}
    dataset_storage_directory: The directory where the dataset will be saved to.
    parallelize: If True, the all available cores will be used.
    verbose: If true, more feedback is printed.

    Returns
    -------
    A list of dictionaries. Each dictionary contains data corresponding to one random
    Hamiltonian with multiple disorder realizations.
    """
    hams = gen_bulk_hamiltonians(
        params=params,
        dataset_path=dataset_path,
        parallelize=parallelize,
        verbose=verbose,
    )
    data = generate_realizations(
        dataset_path=dataset_path, dataset=hams, verbose=verbose
    )
    return data
