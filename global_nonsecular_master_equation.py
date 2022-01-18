import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import plot_utils
import state_perturbation_lib as spl

# Master equation for RWA global master equation with no secular approximation

sigma_z_hot = qt.tensor(qt.sigmaz(), qt.qeye(2))
sigma_z_cold = qt.tensor(qt.qeye(2), qt.sigmaz())

sigma_plus_hot = qt.tensor(qt.sigmap(), qt.qeye(2))
sigma_plus_cold = qt.tensor(qt.qeye(2), qt.sigmap())
sigma_minus_hot = qt.tensor(qt.sigmam(), qt.qeye(2))
sigma_minus_cold = qt.tensor(qt.qeye(2), qt.sigmam())


def mu_plus(epsilon, g, beta, gamma):
    return (gamma(epsilon + g) * (1 + spl.nbe(epsilon + g, beta)) + gamma(epsilon - g) * (1 + spl.nbe(epsilon - g, beta))) / 2


def mu_minus(epsilon, g, beta, gamma):
    return (gamma(epsilon + g) * (1 + spl.nbe(epsilon + g, beta)) - gamma(epsilon - g) * (1 + spl.nbe(epsilon - g, beta))) / 2


def nu_plus(epsilon, g, beta, gamma):
    return (gamma(epsilon + g) * spl.nbe(epsilon + g, beta) + gamma(epsilon - g) * spl.nbe(epsilon - g, beta)) / 2


def nu_minus(epsilon, g, beta, gamma):
    return (gamma(epsilon + g) * spl.nbe(epsilon + g, beta) - gamma(epsilon - g) * spl.nbe(epsilon - g, beta)) / 2


def ham_system(epsilon, g):
    return (
            (epsilon / 2) * (sigma_z_hot + sigma_z_cold)
            + g * (sigma_plus_hot * sigma_minus_cold + sigma_plus_cold * sigma_minus_hot)
    )


def lindblad_dissipator_superoperator(operator):
    return (
            qt.sprepost(operator, operator.conj())
            - 0.5 * (qt.spre(operator.conj() * operator) + qt.spost(operator.conj() * operator))
    )


# gamma_cold and gamma_hot should be lambda expressions which take in a frequency and return the coupling strength to
# that frequency mode in the cold/hot bath respectively
def liouvillian_superoperator(epsilon, g, beta_cold, beta_hot, gamma_cold, gamma_hot):
    return (
            -1j * qt.spre(ham_system(epsilon, g)) + 1j * qt.spost(ham_system(epsilon, g))
            + mu_plus(epsilon, g, beta_cold, gamma_cold) * lindblad_dissipator_superoperator(sigma_minus_cold)
            + nu_plus(epsilon, g, beta_cold, gamma_cold) * lindblad_dissipator_superoperator(sigma_plus_cold)
            - 0.5 * mu_minus(epsilon, g, beta_cold, gamma_cold) * (
                    qt.spre(sigma_plus_cold * sigma_minus_hot)
                    + qt.spost(sigma_plus_hot * sigma_minus_cold)
                    + qt.sprepost(sigma_z_cold * sigma_minus_hot, sigma_plus_cold)
                    + qt.sprepost(sigma_minus_cold, sigma_z_cold * sigma_plus_hot)
            )
            + 0.5 * nu_minus(epsilon, g, beta_cold, gamma_cold) * (
                    qt.spre(sigma_minus_cold * sigma_plus_hot)
                    + qt.spost(sigma_minus_hot * sigma_plus_cold)
                    - qt.sprepost(sigma_z_cold * sigma_plus_hot, sigma_minus_cold)
                    - qt.sprepost(sigma_plus_cold, sigma_z_cold * sigma_minus_hot)
            )

            + mu_plus(epsilon, g, beta_hot, gamma_hot) * lindblad_dissipator_superoperator(sigma_minus_hot)
            + nu_plus(epsilon, g, beta_hot, gamma_hot) * lindblad_dissipator_superoperator(sigma_plus_hot)
            - 0.5 * mu_minus(epsilon, g, beta_hot, gamma_hot) * (
                    qt.spre(sigma_plus_hot * sigma_minus_cold)
                    + qt.spost(sigma_plus_cold * sigma_minus_hot)
                    + qt.sprepost(sigma_z_hot * sigma_minus_cold, sigma_plus_hot)
                    + qt.sprepost(sigma_minus_hot, sigma_z_hot * sigma_plus_cold)
            )
            + 0.5 * nu_minus(epsilon, g, beta_hot, gamma_hot) * (
                    qt.spre(sigma_minus_hot * sigma_plus_cold)
                    + qt.spost(sigma_minus_cold * sigma_plus_hot)
                    - qt.sprepost(sigma_z_hot * sigma_plus_cold, sigma_minus_hot)
                    - qt.sprepost(sigma_plus_hot, sigma_z_hot * sigma_minus_cold)
            )
    )


def calculate_steady_state(epsilon, g, beta_cold, beta_hot, gamma_cold, gamma_hot):
    return qt.steady(liouvillian_superoperator(epsilon, g, beta_cold, beta_hot, gamma_cold, gamma_hot))


def calculate_steady_concurrence(epsilon, g, beta_cold, beta_hot, gamma_cold, gamma_hot):
    return qt.concurrence(calculate_steady_state(epsilon, g, beta_cold, beta_hot, gamma_cold, gamma_hot))


if __name__ == '__main__':
    epsilon = 1.
    g = 0.02
    gamma_hot = lambda x: 0.001
    gamma_cold = lambda x: 0.1
    temperatures_cold = np.array([0.01, 0.2, 0.25, 0.32, 0.4])
    beta_cold_1, beta_cold_2, beta_cold_3, beta_cold_4, beta_cold_5 = 1. / temperatures_cold
    temperature_hot_min = 0.4
    temperature_hot_max = 60.

    plotting_function_1 = lambda temperature_hot: calculate_steady_concurrence(epsilon, g, beta_cold_1, 1./temperature_hot, gamma_cold,
                                                                        gamma_hot)
    plotting_function_2 = lambda temperature_hot: calculate_steady_concurrence(epsilon, g, beta_cold_2, 1./temperature_hot, gamma_cold,
                                                                        gamma_hot)
    plotting_function_3 = lambda temperature_hot: calculate_steady_concurrence(epsilon, g, beta_cold_3, 1./temperature_hot, gamma_cold,
                                                                        gamma_hot)
    plotting_function_4 = lambda temperature_hot: calculate_steady_concurrence(epsilon, g, beta_cold_4, 1./temperature_hot, gamma_cold,
                                                                        gamma_hot)
    plotting_function_5 = lambda temperature_hot: calculate_steady_concurrence(epsilon, g, beta_cold_5, 1./temperature_hot, gamma_cold,
                                                                        gamma_hot)

    plot_points_1 = plot_utils.generate_plot_points(plotting_function_1, temperature_hot_min, temperature_hot_max, 100)
    plot_points_2 = plot_utils.generate_plot_points(plotting_function_2, temperature_hot_min, temperature_hot_max, 100)
    plot_points_3 = plot_utils.generate_plot_points(plotting_function_3, temperature_hot_min, temperature_hot_max, 100)
    plot_points_4 = plot_utils.generate_plot_points(plotting_function_4, temperature_hot_min, temperature_hot_max, 100)
    plot_points_5 = plot_utils.generate_plot_points(plotting_function_5, temperature_hot_min, temperature_hot_max, 100)

    plt.plot(plot_points_1, plot_points_2, plot_points_3, plot_points_4, plot_points_5)
    plt.xlabel('temperature_hot')
    plt.ylabel('concurrence')
    plt.legend(['0.01', '0.2', '0.25', '0.32', '0.4'])
    plt.savefig('results.png')
