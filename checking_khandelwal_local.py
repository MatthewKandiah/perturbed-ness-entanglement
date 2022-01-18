import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import state_perturbation_lib as spl

# Perturbing steady state from Eqn 22 in Khandelwal 2020
from khandelwal_local_state import KhandelwalLocalState

if __name__ == '__main__':
    parameter_dict = {
        'epsilon': 1.,
        'g': 0.0016,
        'gamma_hot': 0.001,
        'gamma_cold': 0.011,
        'temperature_hot': 0.25,
        'temperature_cold': 0.1,
    }

    NUMBER_OF_PERTURBED_STATES_TO_PLOT = 1000

    khandelwal_local_state = KhandelwalLocalState(parameter_dict)

    # we want to perturb each of these terms by a term up to the coupling strength squared
    # to see if the entanglement result survives
    # taking this to be the average coupling strength divided by the system energy scale
    maximum_perturbation = (((khandelwal_local_state.gamma_hot + khandelwal_local_state.gamma_cold) / 2) ** 2
                            / np.sqrt(khandelwal_local_state.epsilon ** 2 + khandelwal_local_state.g ** 2))


    def generate_perturbation_concurrence_plot_point(perturbed_density_operator, original_density_operator):
        return (spl.qobj_absolute_difference(perturbed_density_operator, original_density_operator),
                qt.entropy.concurrence(perturbed_density_operator))


    perturbed_states = spl.generate_perturbed_states(khandelwal_local_state.rho, maximum_perturbation,
                                                     NUMBER_OF_PERTURBED_STATES_TO_PLOT)

    plot_points_perturbation_size = []
    plot_points_concurrence = []
    for state in perturbed_states:
        perturbation_size, concurrence = generate_perturbation_concurrence_plot_point(state, khandelwal_local_state.rho)
        plot_points_perturbation_size.append(perturbation_size)
        plot_points_concurrence.append(concurrence)

    print(f"unperturbed state: {khandelwal_local_state.rho}")
    print(f"trace: {khandelwal_local_state.rho.tr()}")
    print(f"concurrence: {qt.concurrence(khandelwal_local_state.rho)}")
    plt.scatter(plot_points_perturbation_size, plot_points_concurrence)
    plt.xlabel('Total perturbation')
    plt.ylabel('Concurrence')
    plt.savefig('results.png')
