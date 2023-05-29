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
import gc
import os
from time import time as now

# Third party libraries
import numpy as np

# This project
from chern_predictor.data.analysis import estimate_localization_length
from chern_predictor.data.data_generator import gen_dataset
from chern_predictor.data.data_generator_functions import (
    gen_dataset_name_and_experiment_directory,
    print_progress_status,
)
from chern_predictor.figures.fig_num_states_hist import (
    gen_number_of_states_histogram_data,
    make_number_of_states_histogram,
)
from chern_predictor.figures.fig_num_states_plot import (
    gen_number_of_states_plot_data,
    make_number_of_states_plot,
)
from chern_predictor.figures.fig_performance import (
    gen_performance_plot_data,
    make_performance_fig,
)
from chern_predictor.figures.fig_running_average import (
    gen_running_average_plot_data,
    make_running_average_fig,
)
from chern_predictor.figures.fig_setup import gen_setup_fig_data, make_setup_fig
from chern_predictor.figures.fig_summary import make_summary_fig
from chern_predictor.figures.fig_threshold import (
    gen_threshold_plot_data,
    make_certainty_threshold_fig,
)
from chern_predictor.figures.fig_training_curves import (
    gen_training_curve_data,
    make_training_curve_fig,
)
from chern_predictor.networks.evaluation import evaluate_networks_in_ensemble
from chern_predictor.networks.helper_functions import load_data, prepare_folders
from chern_predictor.networks.training import train_ensemble
from default_parameters import evaluation_params, network_parameters, params

if __name__ == "__main__":

    initial_time = now()

    # # # (De)activate parts of the pipeline # # #
    generate_data = True
    analyze_data = True
    train_networks = True
    evaluate_networks = True
    plot_network_figures = True

    # # # Configure size of datasets # # #
    num_training_hamiltonians_per_chern_number = 1250
    num_validation_hamiltonians_per_chern_number = 125
    num_test_hamiltonians_per_chern_number = 125

    # # # Setup directories # # #
    dataset_name, experiment_path = gen_dataset_name_and_experiment_directory(
        params=params,
        storage_directory="experiments/",
        suffix="",
    )
    print(f"The name of the dataset is {dataset_name}\n")
    print("Setting up the dataset directory:\n")
    dataset_directory = os.path.join(experiment_path, "datasets")
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    # # # # Generate the datasets # # #
    if generate_data:
        # Generate training dataset
        start_time = now()
        print("Generating the training dataset:\n")
        params["dataset"][
            "num_hams_per_chern_number"
        ] = num_training_hamiltonians_per_chern_number
        params["dataset"]["smallest_ham_seed"] = 0 * 10**8
        training_dataset = gen_dataset(
            params=params, dataset_path=os.path.join(dataset_directory, "training.json")
        )
        print(f"Generating the training dataset took {round(now() - start_time)}s.\n\n")

        # Generate validation dataset
        start_time = now()
        print("Generating the validation dataset:\n")
        params["dataset"][
            "num_hams_per_chern_number"
        ] = num_validation_hamiltonians_per_chern_number
        params["dataset"]["smallest_ham_seed"] = 1 * 10**8
        validation_dataset = gen_dataset(
            params=params,
            dataset_path=os.path.join(dataset_directory, "validation.json"),
        )
        print(
            f"Generating the validation dataset took {round(now() - start_time)}s.\n\n"
        )

        # Generate test dataset
        start_time = now()
        print("Generating the test dataset:\n")
        params["dataset"][
            "num_hams_per_chern_number"
        ] = num_test_hamiltonians_per_chern_number
        params["dataset"]["smallest_ham_seed"] = 2 * 10**8
        test_dataset = gen_dataset(
            params=params, dataset_path=os.path.join(dataset_directory, "test.json")
        )
        print(f"Generating the test dataset took {round(now() - start_time)}s.\n\n")

        # Delete some unused variables to avoid memory overflow
        del training_dataset
        del validation_dataset
        del test_dataset
        gc.collect()

    for ensemble_name in ["cnn32", "cnn64", "cnn64dl"]:
        print(f"Setting up the directories for ensemble {ensemble_name}:\n")
        (
            dataset_directory,
            ensemble_path,
            model_path,
            plot_path,
            figure_path,
        ) = prepare_folders(
            experiment_directory=experiment_path, ensemble_name=ensemble_name
        )

        if analyze_data:
            # # # Analyse the datasets # # #
            print("Analyzing the datasets:\n")

            start_time = now()
            print("Generating figure 4")
            gen_number_of_states_histogram_data(
                dataset_directory=dataset_directory,
                figure_path=figure_path,
                verbose=True,
            )
            make_number_of_states_histogram(figure_path=figure_path)
            print("Generating figure 5")
            gen_number_of_states_plot_data(
                dataset_directory=dataset_directory,
                figure_path=figure_path,
                verbose=True,
            )
            make_number_of_states_plot(figure_path=figure_path)
            print(f"Generating the figures took {round(now() - start_time, 1)}s.\n")

            # Estimate localization length based on LDOS from non-zero Chern number systems
            non_zero_abs_chern_values = [
                val for val in params["dataset"]["chern_abs_vals"] if val > 0
            ]
            start_time = now()
            data_train, _, data_test = load_data(
                dataset_directory=dataset_directory,
                load_training_data=True,
                load_validation_data=False,
                load_test_data=True,
                verbose=True,
            )
            # To estimate the localization length, consider the average LDOS along a
            # strip with an offset of three sites from the boundary and with an offset
            # of strip_width / 2 from the center.
            print("Training dataset:")
            estimate_localization_length(
                dataset=data_train,
                strip_width=4,
                offset=3,
                relative_gap_min=0.0,
                relative_gap_max=np.inf,
                abs_chern_numbers=non_zero_abs_chern_values,
                relative_disorder_strength_max=np.inf,
                filepath=os.path.join(
                    figure_path, "localization_lengths_training_data"
                ),
                plotting=True,
                verbose=True,
            )

            print("\nTest dataset:")
            estimate_localization_length(
                dataset=data_test,
                strip_width=4,
                offset=3,
                relative_gap_min=0.0,
                relative_gap_max=np.inf,
                abs_chern_numbers=non_zero_abs_chern_values,
                relative_disorder_strength_max=np.inf,
                filepath=os.path.join(figure_path, "localization_lengths_test_data"),
                plotting=True,
                verbose=True,
            )
            print(
                f"Calculating the localization lengths took {round(now() - start_time, 1)}s.\n\n"
            )
            print("All calculations are finished.")

            # Delete some unused variables to avoid memory overflow
            del data_train, data_test
            gc.collect()

        if train_networks:
            # # # Train an ensemble of neural networks # # #
            data_training, data_validation, data_test = load_data(
                dataset_directory=dataset_directory
            )  # Loading the data so learn how many datapoints there are

            # Delete some unused variables to avoid memory overflow
            del data_training, data_validation, data_test
            gc.collect()

            print(f"Training the {ensemble_name} ensemble:\n")
            start_time = now()
            train_ensemble(
                dataset_directory=dataset_directory,
                num_nets=network_parameters["num_networks"],
                network_type=ensemble_name,
                num_epochs=network_parameters["num_epochs"],
                ensemble_path=ensemble_path,
                model_path=model_path,
                plot_path=plot_path,
                verbose=True,
            )
            print(
                f"Training the {ensemble_name} ensemble took {round(now() - start_time)}s."
            )

        if evaluate_networks:
            # # # Evaluate the neural networks on the data # # #
            print("Evaluating the neural network ensembles:\n")
            start_time = now()
            evaluate_networks_in_ensemble(
                dataset_directory=dataset_directory,
                ensemble_path=ensemble_path,
                model_path=model_path,
                verbose=True,
            )
            print(
                f"Evaluating the neural network ensembles took {round(now() - start_time)}s."
            )

        if plot_network_figures:
            # # # Make figures # # #
            print("Generating the figures:\n")
            start_time = now()

            # Generate a seed so that the same ensembles are used in all figures
            ensemble_seed = evaluation_params["ensemble_seed"]
            print(f"The ensemble seed is {ensemble_seed}.\n")

            print("Generating figure 1")
            gen_setup_fig_data(dataset_path=dataset_directory, figure_path=figure_path)
            make_setup_fig(figure_path=figure_path)
            print("Generating figure 2")
            gen_performance_plot_data(
                ensemble_path=ensemble_path,
                ensemble_seed=ensemble_seed,
                model_path=model_path,
                dataset_directory=dataset_directory,
                figure_path=figure_path,
            )
            make_performance_fig(figure_path=figure_path, verbose=True)
            print("Generating figure 3")
            gen_threshold_plot_data(
                ensemble_path=ensemble_path,
                ensemble_seed=ensemble_seed,
                model_path=model_path,
                dataset_directory=dataset_directory,
                figure_path=figure_path,
                verbose=True,
            )
            make_certainty_threshold_fig(figure_path=figure_path)
            print("Generating the summary figure")
            make_summary_fig(figure_path=figure_path, verbose=True)
            print("Generating the running average figure")
            gen_running_average_plot_data(
                ensemble_path=ensemble_path,
                model_path=model_path,
                dataset_directory=dataset_directory,
                figure_path=figure_path,
            )
            make_running_average_fig(figure_path=figure_path, verbose=True)
            print("Generating the training curves")
            gen_training_curve_data(
                ensemble_path=ensemble_path,
                model_path=model_path,
                figure_path=figure_path,
            )
            make_training_curve_fig(figure_path=figure_path)
            print(f"\nGenerating the figures took {round(now() - start_time)}s.")

    print(f"\n All experiments are completed.")
    print_progress_status(elapsed_time=now() - initial_time, progress=1.0)
