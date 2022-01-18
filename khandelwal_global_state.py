import qutip as qt
import state_perturbation_lib as spl


# TODO: double check this expression & their parameter definitions, the plot I'm generating
#  doesn't match what I expected
class KhandelwalGlobalState:

    def __init__(self, parameter_dict):
        self.epsilon = parameter_dict['epsilon']
        self.g = parameter_dict['g']
        self.epsilon_plus = self.epsilon + self.g
        self.epsilon_minus = self.epsilon - self.g
        self.gamma_hot = parameter_dict['gamma_hot']
        self.gamma_cold = parameter_dict['gamma_cold']
        self.temperature_hot = parameter_dict['temperature_hot']
        self.temperature_cold = parameter_dict['temperature_cold']
        self.beta_hot = 1. / self.temperature_hot
        self.beta_cold = 1. / self.temperature_cold

        def gamma_hot_plus(energy):
            return self.gamma_hot * spl.nbe(energy, self.beta_hot)

        def gamma_hot_minus(energy):
            return self.gamma_hot * (1 + spl.nbe(energy, self.beta_hot))

        def gamma_cold_plus(energy):
            return self.gamma_cold * spl.nbe(energy, self.beta_cold)

        def gamma_cold_minus(energy):
            return self.gamma_cold * (1 + spl.nbe(energy, self.beta_cold))

        def Gamma_plus(energy):
            return gamma_hot_plus(energy) + gamma_cold_plus(energy)

        def Gamma_minus(energy):
            return gamma_hot_minus(energy) + gamma_cold_minus(energy)

        def Gamma(energy):
            return Gamma_plus(energy) + Gamma_minus(energy)

        self.chi = Gamma(self.epsilon_minus) * Gamma(self.epsilon_plus)

        # steady state from Khandelwal 2020 Eqn (22)
        s1 = (Gamma_plus(self.epsilon_minus) * Gamma_plus(self.epsilon_plus)) / self.chi

        s2 = (Gamma_plus(self.epsilon_minus) * Gamma_minus(self.epsilon_plus) + Gamma_minus(self.epsilon_minus)
              * Gamma_plus(self.epsilon_plus)) / (2 * self.chi)

        s3 = (Gamma_plus(self.epsilon_minus) * Gamma_minus(self.epsilon_plus) + Gamma_minus(self.epsilon_minus)
              * Gamma_plus(self.epsilon_plus)) / (2 * self.chi)

        s4 = (Gamma_minus(self.epsilon_minus) * Gamma_minus(self.epsilon_plus)) / self.chi

        d = (Gamma_minus(self.epsilon_minus) * Gamma_plus(self.epsilon_plus) - Gamma_plus(self.epsilon_minus)
             * Gamma_minus(self.epsilon_plus)) / (2 * self.chi)

        self.rho = qt.Qobj([
            [s1, 0, 0, 0],
            [0, s2, d, 0],
            [0, d.conjugate(), s3, 0],
            [0, 0, 0, s4]
        ], dims=[[2, 2], [2, 2]])
