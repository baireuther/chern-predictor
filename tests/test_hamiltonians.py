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
from math import ceil

# Third party libraries
import numpy as np
import pytest

# This project
from chern_predictor.data.hamiltonian_coefficients import generate_ham_coefficients
from chern_predictor.data.hamiltonian_coefficients import hopping_function as fhop
from chern_predictor.data.ldos import make_square

# Generate a random set of Hamiltonian parameters
hopping_scale = 1.0
hopping_cutoff = 1 + 2 * np.random.rand()
max_relative_pairing_strength = 0.2
seed = np.random.randint(1e8)
Xi, Delta, _ = generate_ham_coefficients(
    hopping_scale,
    hopping_cutoff,
    max_relative_pairing_strength,
    np.random.RandomState(seed),
)


@pytest.mark.parametrize("Xi", [Xi])
@pytest.mark.parametrize("Delta", [Delta])
def test__syms(Xi, Delta):
    """Test if the symmetries for the hoppings and pairings are fulfilled"""
    N = Xi.shape[0] // 2

    def fhop_reduced(dx, dy):
        return fhop(dx, dy, Xi, Delta)

    for dx in range(-N, N + 1):
        for dy in range(-N, N + 1):
            # diagonal terms
            np.testing.assert_almost_equal(
                fhop_reduced(dx, -dy)[0, 0], fhop_reduced(dx, dy)[0, 0]
            )
            np.testing.assert_almost_equal(
                fhop_reduced(-dx, dy)[0, 0], fhop_reduced(dx, dy)[0, 0]
            )
            np.testing.assert_almost_equal(
                fhop_reduced(-dx, -dy)[0, 0], fhop_reduced(dx, dy)[0, 0]
            )
            np.testing.assert_almost_equal(
                fhop_reduced(dx, dy)[0, 0], fhop_reduced(dy, dx)[0, 0]
            )
            np.testing.assert_almost_equal(
                fhop_reduced(dx, dy)[0, 0], -fhop_reduced(dx, dy)[1, 1]
            )
            # off-diagonal terms
            np.testing.assert_almost_equal(
                fhop_reduced(-dx, dy)[0, 1], np.conj(fhop_reduced(dx, dy)[0, 1])
            )
            np.testing.assert_almost_equal(
                fhop_reduced(dx, -dy)[0, 1], -np.conj(fhop_reduced(dx, dy)[0, 1])
            )
            np.testing.assert_almost_equal(
                fhop_reduced(-dx, -dy)[0, 1], -fhop_reduced(dx, dy)[0, 1]
            )
            np.testing.assert_almost_equal(
                fhop_reduced(dy, dx)[0, 1], -1j * np.conj(fhop_reduced(dx, dy)[0, 1])
            )
            np.testing.assert_almost_equal(
                fhop_reduced(dx, dy)[0, 1], np.conj(fhop_reduced(-dx, -dy)[1, 0])
            )


def test__amplitudes(N=1000, margin=0.05):
    """Test if the hopping and pairing amplitudes a) obey the weak pairing criterion and b) the
    exponential decay."""
    hopping_cutoff = 2.0
    hopping_scale = 1.5
    max_relative_pairing_strength = 0.5
    hopping_cutoff_rounded_up = int(ceil(hopping_cutoff))
    xis = np.zeros(
        shape=(N, 2 * hopping_cutoff_rounded_up + 1, 2 * hopping_cutoff_rounded_up + 1),
        dtype=float,
    )
    deltas = np.zeros(
        shape=(N, 2 * hopping_cutoff_rounded_up + 1, 2 * hopping_cutoff_rounded_up + 1),
        dtype=complex,
    )
    relative_pairing_strengths = np.zeros(shape=(N, 1), dtype=float)
    for n, seed in enumerate(range(1000)):
        xi, delta, relative_pairing_strength = generate_ham_coefficients(
            hopping_scale,
            hopping_cutoff,
            max_relative_pairing_strength,
            np.random.RandomState(seed),
        )
        xis[n] = xi
        deltas[n] = delta
        relative_pairing_strengths[n] = relative_pairing_strength

    # Test relative pairing strengths are below pairing factor
    assert np.all(
        max_relative_pairing_strength**2
        * (np.sum(np.abs(xis) ** 2, axis=(1, 2)) - np.abs(xis[:, 0, 0]) ** 2)
        >= np.sum(np.abs(deltas) ** 2, axis=(1, 2))
    )
    rps = np.sqrt(
        np.sum(np.abs(deltas) ** 2, axis=(1, 2))
        / (np.sum(np.abs(xis) ** 2, axis=(1, 2)) - np.abs(xis[:, 0, 0]) ** 2)
    ).reshape(-1, 1)
    assert np.all(rps == relative_pairing_strengths)
    assert np.max(relative_pairing_strengths) <= max_relative_pairing_strength
    assert np.min(relative_pairing_strengths) >= 0

    # Test hopping amplitudes
    for nx in range(-hopping_cutoff_rounded_up, hopping_cutoff_rounded_up + 1):
        for ny in range(-hopping_cutoff_rounded_up, hopping_cutoff_rounded_up + 1):
            xi_vals = xis[:, nx, ny]
            delta_vals = np.abs(deltas[:, nx, ny])
            dist = np.sqrt(nx**2 + ny**2)
            if dist > hopping_cutoff:
                assert np.max(np.abs(xi_vals)) == 0
                assert np.max(np.abs(delta_vals)) == 0
            elif dist > 0:
                factor = np.exp(-dist / hopping_scale)
                assert -factor <= np.min(xi_vals) < -factor + margin
                assert factor - margin < np.max(xi_vals) <= factor
                assert np.max(delta_vals) <= factor
                # For the pairing, it is only checked that the moduli are not too large. The
                # largest moduli can be a bit smaller than `factor` due to the weak pairing
                # condition.


@pytest.mark.parametrize("xi", [Xi])
@pytest.mark.parametrize("delta", [Delta])
@pytest.mark.parametrize("hopping_cutoff", [hopping_cutoff])
def test__matrix_elements_2d(xi, delta, hopping_cutoff, edge_length=16):
    """Build and test the tight binding 2D Hamiltonian."""

    ham0 = make_square(
        edge_length=edge_length,
        hopping_cutoff=hopping_cutoff,
        xi=xi,
        delta=delta,
        sparse=False,
    )
    ham0 = ham0.reshape(edge_length**2, 2, edge_length**2, 2).transpose(0, 2, 1, 3)

    hopping_cutoff_rounded_up = int(ceil(hopping_cutoff))
    for mx in range(hopping_cutoff_rounded_up + 1):
        for my in range(hopping_cutoff_rounded_up + 1):
            for nx in range(edge_length - mx):
                for ny in range(edge_length - my):
                    # from site (nx, ny) to site (nx + mx, ny + my)
                    from_site = nx * edge_length + ny
                    to_site = (nx + mx) * edge_length + (ny + my)
                    assert xi[mx, my] == xi[-mx, -my]
                    assert ham0[to_site][from_site][0][0] == xi[mx, my]
                    assert ham0[to_site][from_site][1][1] == -xi[-mx, -my]
                    assert ham0[to_site][from_site][0][1] == delta[mx, my]
                    assert ham0[to_site][from_site][1][0] == np.conj(delta[-mx, -my])

                    # from site n+m to site n
                    assert ham0[from_site][to_site][0][0] == xi[-mx, -my]
                    assert ham0[from_site][to_site][1][1] == -xi[mx, my]
                    assert ham0[from_site][to_site][0][1] == delta[-mx, -my]
                    assert ham0[from_site][to_site][1][0] == np.conj(delta[mx, my])
