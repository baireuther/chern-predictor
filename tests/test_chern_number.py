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


# Third party libraries
import numpy as np
import pytest

# This project
from chern_predictor.data.chern_numbers import calc_chern_number
from chern_predictor.data.hamiltonian_coefficients import generate_ham_coefficients
from chern_predictor.data.spectrum import calc_gap


@pytest.mark.parametrize("seed", np.random.randint(0, 1e8, 10))
def test__chern_number_is_one(seed):
    """Tests if the calculation of the Chern number is correct by a scenario in which the Chern
    number is always +-1."""

    # If the gap is very small, the calculation of the Chern number can take very long. To keep
    # the test's runtime small, only Hamiltonians with absolute_gap > 0.1 are considered.
    absolute_gap = 0
    while absolute_gap < 0.1:
        # First, generate the hoppings and pairings
        Xi, Delta, _ = generate_ham_coefficients(
            hopping_scale=1e6,
            hopping_cutoff=1.0,
            max_relative_pairing_strength=0.2,
            rng=np.random.RandomState(seed),
        )
        Xi = np.sign(Xi)

        # Calculate the gap
        relative_gap, absolute_gap = calc_gap(
            xi=Xi,
            delta=Delta,
            seed=seed,
            min_absolute_gap=0.1,
            verbose=False,
        )
        seed = np.random.randint(0, 1e8, 10)

    # Then evaluate the Chern number
    chern_no = calc_chern_number(Xi, Delta)
    assert chern_no == 1 or chern_no == -1


@pytest.mark.parametrize("seed", np.random.randint(0, 1e8, 10))
def test__chern_number_is_zero(seed):
    """Tests if the calcualtion of the Chern number is correct by the scenario where the
    chemical potential is larger than the bandwidth and the Chern number hence zero."""

    # If the gap is very small, the calculation of the Chern number can take very long. To keep
    # the test's runtime small, only Hamiltonians with absolute_gap > 0.1 are considered.
    absolute_gap = 0
    while absolute_gap < 0.1:
        # First, generate the hoppings and pairings
        Xi, Delta, _ = generate_ham_coefficients(
            hopping_scale=1e6,
            hopping_cutoff=1.0,
            max_relative_pairing_strength=0.2,
            rng=np.random.RandomState(seed),
        )
        Xi[0, 0] = 10.0

        # Calculate the gap
        relative_gap, absolute_gap = calc_gap(
            xi=Xi,
            delta=Delta,
            seed=seed,
            min_absolute_gap=0.1,
            verbose=False,
        )
        seed = np.random.randint(0, 1e8, 10)

    # Then evaluate the Chern number
    chern_no = calc_chern_number(Xi, Delta)
    assert chern_no == 0
