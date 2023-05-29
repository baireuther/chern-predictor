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
import subprocess

# This project
from chern_predictor.data import __version__

# Bulk Hamiltonian parameters
bulk_ham_params = {
    "hopping_cutoff": 2,
    "hopping_decay_scale": 1.0,
    "max_relative_pairing_strength": 1.0,
    "min_absolute_gap": 3.5 / 24.0,  # 3.5 over edge-length
}

# System parameters
system_params = {
    "num_sites_x": 24,
    "num_sites_y": 24,
    "geometry": "rectangle with hard wall boundary condition",
}

# Dataset parameters
dataset_params = {
    "chern_abs_vals": [0, 1, 2, 3],
    "ldos_window_as_fraction_of_gap": 1 / 3.0,
    "relative_disorder_strengths": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "smallest_ham_seed": 1,
    "num_hams_per_chern_number": int(1),
    "version": __version__,
}

try:
    dataset_params["git_hash"] = str(
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    )
except:
    print("WARNING: git-hash not found.")

data_generation_params = {"chunk_size": 2000}

params = {
    "bulk_ham": bulk_ham_params,
    "system": system_params,
    "dataset": dataset_params,
    "data_generation": data_generation_params,
}

# Network parameters
network_parameters = {"num_networks": 13, "num_epochs": 50}

# Evaluation parameters
evaluation_params = {
    "ensemble_seed": 0,
    "ensemble_sizes": [1, 2, 4, 8],
    "num_bootstraps": 13,
    "colors": ["gray", "tab:blue", "tab:orange", "tab:red"],
    "epochs_until_trained": 20,
}
