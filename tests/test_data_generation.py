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
import warnings

# Third party libraries
import numpy as np
import pytest
from scipy import sparse

# This project
from chern_predictor.data.data_generator import gen_dataset
from chern_predictor.data.hamiltonian_coefficients import generate_ham_coefficients
from chern_predictor.data.ldos import calc_ldos_sparse, make_square
from default_parameters import params


def assert_almost_equal(e1, e2, tolerance):
    if type(e1) == dict:
        assert_dict_almost_equal(e1, e2, tolerance)
    elif type(e1) == list:
        assert len(e1) == len(e2)
        for n in range(len(e1)):
            assert_almost_equal(e1[n], e2[n], tolerance)
    elif type(e1) == np.ndarray:
        np.testing.assert_array_almost_equal(e1, e2, decimal=tolerance)
    else:
        try:
            assert abs(e1 - e2) < tolerance
        except:
            assert e1 == e2


def assert_dict_almost_equal(d1: dict, d2: dict, tolerance: float):
    ks = sorted(d1.keys())
    assert ks == sorted(d2.keys())
    for k in ks:
        e1 = d1[k]
        e2 = d2[k]
        if k == "chunk_size":
            if e1 != e2:
                print(
                    f"Notice: The `chunk_sizes` are different, namely {e1} and {e2}, however "
                    f"since this has no effect on the dataset it is ignored."
                )
        else:
            assert_almost_equal(e1, e2, tolerance)


def test_functions_to_compare_dicts(
    reference_fname="resources/dataset_reference.json",
):
    try:
        with open(reference_fname, "r") as file:
            d1 = json.load(file)
    except:
        # The CI/CD pipeline seems to execute tests differently
        with open("tests/" + reference_fname, "r") as file:
            d1 = json.load(file)
    d2 = copy.deepcopy(d1)

    d1[1]["local_density_of_states"][6] += 1e-8

    assert not d1 == d2
    assert_almost_equal(d1, d2, tolerance=1e-7)
    with pytest.raises(Exception):
        assert_almost_equal(d1, d2, tolerance=1e-9)


@pytest.mark.parametrize("seed", np.random.randint(0, 1e8, size=1))
def test_dataset_reproducibility(seed):

    print("seed", seed)
    params["dataset"]["smallest_ham_seed"] = int(seed)
    params["data_generation"]["chunk_size"] = 20
    dataset = gen_dataset(params)

    # Use serial data generation to demonstrate that the chunk_size has no impact on the dataset.
    params["data_generation"]["chunk_size"] = 1
    same_dataset = gen_dataset(params)
    for dat in same_dataset:
        dat["data_generation_params"][
            "chunk_size"
        ] = 20  # Setting the `chunk_size` to the same
        # value as in the reference dataset
    for n in range(len(dataset)):
        if dataset[n] != same_dataset[n]:
            warnings.warn(
                f"The datapoint number {n} is not equivalent. Checking if there are "
                f"almost equal next."
            )
            tol = 1e-8
            assert_almost_equal(dataset[n], same_dataset[n], tol)
            warnings.warn(
                f"The datapoints were indeed almost equal with a tolerance of {tol}."
            )


def test_dataset_algorithm_has_not_changed(
    reference_fname="resources/dataset_reference.json",
):

    try:
        with open(reference_fname, "r") as file:
            dataset_reference = json.load(file)
    except:
        # The CI/CD pipeline seems to execute tests differently
        with open("tests/" + reference_fname, "r") as file:
            dataset_reference = json.load(file)

    datapoint = dataset_reference[0]
    params["bulk_hams"] = datapoint["bulk_ham_params"]
    params["system"] = datapoint["system_params"]
    params["dataset"] = datapoint["dataset_params"]
    params["data_generation"] = datapoint["data_generation_params"]
    params["data_generation"]["chunk_size"] = 20
    dataset = gen_dataset(params)

    for n in range(len(dataset_reference)):
        if dataset[n] != dataset_reference[n]:
            warnings.warn(
                f"The datapoint number {n} is not equivalent. Checking if there are "
                f"almost equal next."
            )
            tol = 1e-8
            assert_almost_equal(dataset[n], dataset_reference[n], tol)
            warnings.warn(
                f"The datapoints were indeed almost equal with a tolerance of {tol}."
            )


@pytest.mark.parametrize("seed", np.random.randint(0, 1e8, size=1))
def test_ldos_calc_is_reducible(seed):

    # Parameters
    edge_length = 24
    hopping_scale = 1.0
    hopping_cutoff = 2.0

    # First, generate the hoppings and pairings
    xi, delta, _ = generate_ham_coefficients(
        hopping_scale=hopping_scale,
        hopping_cutoff=hopping_cutoff,
        max_relative_pairing_strength=0.2,
        rng=np.random.RandomState(seed),
    )

    ham0 = make_square(
        edge_length=edge_length, hopping_cutoff=hopping_cutoff, xi=xi, delta=delta
    )

    rng = np.random.RandomState(seed=seed)
    for relative_disorder_strength in [0.9]:
        disorder_seed = rng.randint(0, 10**9)
        rng_local = np.random.RandomState(disorder_seed)

        # Generate disordered Hamiltonian
        U0 = relative_disorder_strength * 0.2
        u = U0 * (1 - 2 * rng_local.rand(edge_length * edge_length))
        u = u.reshape(-1, 1) * np.array([1, -1]).reshape(1, 2)
        u = u.flatten()
        u_sparse = sparse.diags(u)
        ham = ham0 + u_sparse

        # Calculate LDOS
        evals, ldos = calc_ldos_sparse(
            copy.deepcopy(copy.deepcopy(ham)),
            en_lim=0.3 * (0.2 / 2.0),
            en_resolved=False,
            kmin=24,
            seed=copy.deepcopy(disorder_seed),
            verbose=True,
        )

        # Calculate LDOS
        same_evals, same_ldos = calc_ldos_sparse(
            copy.deepcopy(copy.deepcopy(ham)),
            en_lim=0.3 * (0.2 / 2.0),
            en_resolved=False,
            kmin=24,
            seed=copy.deepcopy(disorder_seed),
            verbose=True,
        )

        for e1, e2 in zip(evals, same_evals):
            if e1 != e2:
                print(e1, e2)
            assert e1 == e2
        assert np.all(evals == same_evals)

        for e1, e2 in zip(ldos, same_ldos):
            if e1 != e2:
                print(e1, e2)
            assert e1 == e2
        assert np.all(ldos == same_ldos)
