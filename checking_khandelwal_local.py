import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import state_perturbation_lib as spl

# Perturbing steady state from Eqn 22 in Khandelwal 2020
from khandelwal_local_state import KhandelwalLocalState

if __name__ == '__main__':
    epsilon = 1.
    g = 0.0016
    gamma_hot = 0.001
    gamma_cold = 0.011
    temperature_hot = 0.25
    temperature_cold = 0.1
    beta_hot = 1. / temperature_hot
    beta_cold = 1. / temperature_cold

    parameter_dict = {
        'epsilon': epsilon,
        'g': g,
        'gamma_hot': gamma_hot,
        'gamma_cold': gamma_cold,
        'temperature_hot': temperature_hot,
        'temperature_cold': temperature_cold,
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

    # check steady state is actually steady state by subbing it into the master equation, expect to get zero
    sigma_z_hot = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sigma_z_cold = qt.tensor(qt.qeye(2), qt.sigmaz())
    sigma_plus_hot = qt.tensor(qt.sigmap(), qt.qeye(2))
    sigma_plus_cold = qt.tensor(qt.qeye(2), qt.sigmap())
    sigma_minus_hot = qt.tensor(qt.sigmam(), qt.qeye(2))
    sigma_minus_cold = qt.tensor(qt.qeye(2), qt.sigmam())

    HamS = (epsilon * (sigma_plus_cold * sigma_minus_cold + sigma_plus_hot * sigma_minus_hot)
            + g * (sigma_plus_hot * sigma_minus_cold + sigma_minus_hot * sigma_plus_cold))

    lindblad_dissipator_superoperator = lambda operator: (
            qt.sprepost(operator, operator.conj())
            - 0.5 * (qt.spre(operator.conj() * operator) + qt.spost(operator.conj() * operator))
    )

    local_liouvillian_superoperator = (-1j * (qt.spre(HamS) - qt.spost(HamS))
                                       + khandelwal_local_state.gamma_hot_plus * lindblad_dissipator_superoperator(sigma_plus_hot)
                                       + khandelwal_local_state.gamma_hot_minus * lindblad_dissipator_superoperator(sigma_minus_hot)
                                       + khandelwal_local_state.gamma_cold_plus * lindblad_dissipator_superoperator(sigma_plus_cold)
                                       + khandelwal_local_state.gamma_cold_minus * lindblad_dissipator_superoperator(sigma_minus_cold)
                                       )

    # print(f"unperturbed state: {khandelwal_local_state.rho}")
    print(f"trace: {khandelwal_local_state.rho.tr()}")
    print(f"concurrence: {qt.concurrence(khandelwal_local_state.rho)}")
    print(f"Check solution is steady state - expect zero: \n{local_liouvillian_superoperator*qt.operator_to_vector(khandelwal_local_state.rho)}")
    plt.scatter(plot_points_perturbation_size, plot_points_concurrence)
    plt.xlabel('Total perturbation')
    plt.ylabel('Concurrence')
    plt.savefig('results.png')
