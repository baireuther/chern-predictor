# Chern predictor
This is a software project by Paul Baireuther.
It is displayed here to allow researchers to understand how the results in reference [[1]](#reference) were obtained and to inspect all details of the implementation.

## Table of contents
[Description](#description)</br>
[Reference](#reference)</br>
[Dataset format](#dataset-format)</br>
[License information](#license-information)

## Description
There are three main aspects to this software:</br>

1) Generation of training data from a distribution over random Hamiltonians, discretized on a lattice including onsite disorder.

2) Training of neural networks to predict the Chern number based on local density of states data.

3) Combination of these neural networks into ensembles to make collective predictions about the Chern number.

<img src=chern_predictor/figures/drawings/illustration.png alt="drawing" width="550"/>

## Test dataset
The Shiba lattice model dataset, used as test dataset in [[1]](#reference), was generated using a different software which is not part of this repository.
In order to generate the results in [[1]](#reference), the automatically generated `test.dat` dataset was replaced by said Shiba dataset.

## Reference
[1] P. Baireuther, M. Płodzień, T. Ojanen, J. Tworzydło, T. Hyart, "Identifying Chern numbers of superconductors from local measurements", https://arxiv.org/abs/2112.06777.

## Dataset format
The datasets are structured as list of datapoints in json format. Each datapoint is a dictionary.
### Complete set of entries
<ul>
<li> bulk_ham_params: (dictionary)</li>
    <ul>
    <li>hopping_cutoff: Maximal hopping distance to be considered (float)</li>
    <li>hopping_decay_scale: Scale on which hopping amplitude decays exponentially (float)</li>
    <li>max_relative_pairing_strength: Maximum ratio of pairing amplitude to hopping 
amplitude to be considered (float)</li>
    <li>min_absolute_gap: Smallest gapsize to be considered in absolute units (float)</li>
    </ul>
<li>system_params: (dictionary)</li>
    <ul>
    <li>num_sites_x: Number of sites in x-direction (integer)</li>
    <li>num_sites_y: Number of sites in y-direction (integer)</li>
    <li>geometry: Description of the lattice shape in words (string)</li>
    </ul>
<li>dataset_params: (dictionary)</li>
    <ul>
    <li>chern_abs_vals: Set of Chern number moduli in the dataset (list of integers)</li>
    <li>ldos_window_as_fraction_of_gap: Size of the energy window to be considered in the LDOS 
calculation in units of the gap-size (float)</li>
    <li>relative_disorder_strengths: Set of disorder amplitudes in units of the gap-size (list 
of floats)</li>
    <li>smallest_ham_seed: Smallest seed, after that consecutively increasing seeds are used 
(integer)
</li>
    <li>num_hams_per_chern_number: Number of Hamiltonians to be generated per value of the 
modulus of the Chern number (integer)</li>
    <li>version: Version of the software (x.x.x)</li>
    <li>git_hash: The hash of the git-version of the code used to generate the data (hash 
number)</li>
    <li>average_gap: Gap-size averaged over all Hamiltonians of the clean and 
infinite systems (float)</li>
    <li>average_bandwidth: Bandwidth averaged over all Hamiltonians of the clean and 
infinite systems (float)</li>
    </ul>
<li>data_generation_params: (dictionary)</li>
    <ul>
    <li>chunk_size Chunk size used in parallel generation of Hamiltonians (integer)</li>
    </ul>
<li>ham_seed: Seed used in generation of bulk Hamiltonian (integer)</li>
<li>relative_pairing_strength: Strength of the pairing relative to the hopping amplitudes 
(float)</li>
<li>absolute_gap: Size of the bulk gap in absolute unites (float)</li>
<li>relative_gap: Size of the bulk gap in units of the bandwidth (float)</li>
<li>bandwidth: Bandwidth in absolute units (float)</li>
<li>chern_number: Chern number of bulk Hamiltonian (integer)</li>
<li>chern_number_absolute_value: Modulus of the Chern number (integer)</li>
<li>disorder_seed: Seed used in generation of (disordered) system realization (integer)</li>
<li>relative_disorder_strength: Disorder strength in units of the average band gap in the 
dataset (float)</li>
<li>eigenvalues_in_ldos: Eigenvalues of states that form the LDOS (list of floats)</li>
<li>num_states_in_ldos: Number of states that form the LDOS (integer)</li>
<li>local_density_of_states: Local density of states (list of floats) with the following format

     # The order of sites in terms of (x, y) in this list is
     # ( 0, 0), ( 0, 1), ..., ( 0, 23), 
     # ( 1, 0), ( 1, 1), ..., ( 1, 23),
     #   ...
     # (23, 0), (23, 1), ..., (23, 23)
</li>
</ul>

### Minimal set of entries
<ul>
<li>system_params: (dictionary)</li>
    <ul>
    <li>num_sites_x: Number of sites in x-direction (integer)</li>
    <li>num_sites_y: Number of sites in y-direction (integer)</li>
    </ul>
<li>relative_gap: Size of the bulk gap in units of the bandwidth (float)</li>
<li>chern_number_absolute_value: Modulus of the Chern number (integer)</li>
<li>relative_disorder_strength: Disorder strength in units of the average band gap in the 
dataset (float)</li>
<li>num_states_in_ldos: Number of states that form the LDOS (integer)</li>
<li>local_density_of_states: Local density of states (list of floats)</li>
</ul>

## License information
Copyright (c) 2020-2023, Paul Baireuther</br>
All rights reserved.

This is a software project by Paul Baireuther.
For the avoidance of doubt: Author and copyright holder of this software is Paul Baireuther who is publishing it as a **private activity** and not in the name of his affiliation(s). 

Its purpose is to allow researchers to understand how the results in reference [[1]](#reference) were obtained and to inspect the details of the implementation.

Currently, this software is not available for (re)use.
