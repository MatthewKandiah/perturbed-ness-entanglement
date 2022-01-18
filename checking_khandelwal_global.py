import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import state_perturbation_lib as spl

# Perturbing steady state from Eqn 22 in Khandelwal 2020
from khandelwal_global_state import KhandelwalGlobalState

if __name__ == '__main__':
    parameter_dict = {
        'epsilon': 1.,
        'g': 0.74,
        'gamma_hot': 0.01,
        'gamma_cold': 0.01,
        'temperature_hot': 1.0,
        'temperature_cold': 0.01,
    }

    NUMBER_OF_PERTURBED_STATES_TO_PLOT = 1000

    khandelwal_global_state = KhandelwalGlobalState(parameter_dict)

    # we want to perturb each of these terms by a term up to the coupling strength squared
    # to see if the entanglement result survives
    # taking this to be the average coupling strength divided by the system energy scale
    maximum_perturbation = (((khandelwal_global_state.gamma_hot + khandelwal_global_state.gamma_cold) / 2) ** 2
                            / np.sqrt(khandelwal_global_state.epsilon ** 2 + khandelwal_global_state.g ** 2))


    def generate_perturbation_concurrence_plot_point(perturbed_density_operator, original_density_operator):
        return (spl.qobj_absolute_difference(perturbed_density_operator, original_density_operator),
                qt.entropy.concurrence(perturbed_density_operator))


    perturbed_states = spl.generate_perturbed_states(khandelwal_global_state.rho, maximum_perturbation,
                                                     NUMBER_OF_PERTURBED_STATES_TO_PLOT)

    plot_points_perturbation_size = []
    plot_points_concurrence = []
    for state in perturbed_states:
        perturbation_size, concurrence = generate_perturbation_concurrence_plot_point(state, khandelwal_global_state.rho)
        plot_points_perturbation_size.append(perturbation_size)
        plot_points_concurrence.append(concurrence)

    print(f"unperturbed state: {khandelwal_global_state.rho}")
    print(f"trace: {khandelwal_global_state.rho.tr()}")
    print(f"concurrence: {qt.concurrence(khandelwal_global_state.rho)}")
    plt.scatter(plot_points_perturbation_size, plot_points_concurrence)
    plt.xlabel('Total perturbation')
    plt.ylabel('Concurrence')
    plt.savefig('results.png')
