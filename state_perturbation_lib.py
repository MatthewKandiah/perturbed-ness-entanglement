import random
import numpy as np
import qutip as qt


def randomly_perturb_populations(density_operator, maximum_perturbation):
    matrix_dimensions = density_operator.shape

    # require 4x4 square matrix
    if matrix_dimensions != (4, 4):
        raise Exception(
            f'2 qubit density matrix must be 4x4, actual dimensions were {matrix_dimensions[0]}x{matrix_dimensions[1]}')

    def perturbation():
        return random.uniform(-maximum_perturbation, maximum_perturbation)

    perturbation = qt.Qobj([
        [perturbation(), 0, 0, 0],
        [0, perturbation(), 0, 0],
        [0, 0, perturbation(), 0],
        [0, 0, 0, perturbation()]
    ], dims=[[2, 2], [2, 2]])

    return density_operator + perturbation


def generate_perturbed_states(density_operator, maximum_perturbation, number_of_states):
    result = []
    for n in range(0, number_of_states):
        result.append(randomly_perturb_populations(density_operator, maximum_perturbation))
    return result


def matrix_absolute_difference(m1, m2):
    total = 0
    for i in range(0, len(m1)):
        for j in range(0, len(m1[0])):
            total += abs(m1[i][j] - m2[i][j])
    return total


def qobj_absolute_difference(q1, q2):
    return matrix_absolute_difference(q1.full(), q2.full())


def nbe(energy, inverse_temp):
    return 1. / (np.exp(inverse_temp * energy) - 1)
