import qutip as qt
import state_perturbation_lib as spl


class KhandelwalLocalState:

    def __init__(self, parameter_dict):
        self.epsilon = parameter_dict['epsilon']
        self.g = parameter_dict['g']
        self.gamma_hot = parameter_dict['gamma_hot']
        self.gamma_cold = parameter_dict['gamma_cold']
        self.temperature_hot = parameter_dict['temperature_hot']
        self.temperature_cold = parameter_dict['temperature_cold']
        self.beta_hot = 1. / self.temperature_hot
        self.beta_cold = 1. / self.temperature_cold

        self.gamma_hot_plus = self.gamma_hot * spl.nbe(self.epsilon, self.beta_hot)
        self.gamma_hot_minus = self.gamma_hot * (1 + spl.nbe(self.epsilon, self.beta_hot))
        self.gamma_cold_plus = self.gamma_cold * spl.nbe(self.epsilon, self.beta_cold)
        self.gamma_cold_minus = self.gamma_cold * (1 + spl.nbe(self.epsilon, self.beta_cold))
        self.Gamma_hot = self.gamma_hot_plus + self.gamma_hot_minus
        self.Gamma_cold = self.gamma_cold_plus + self.gamma_cold_minus
        self.Gamma = self.Gamma_hot + self.Gamma_cold
        self.chi = (4 * (self.g ** 2) + self.Gamma_hot * self.Gamma_cold) * (self.Gamma ** 2)

        # steady state from Khandelwal 2020 Eqn (6)
        r1 = (4 * self.g ** 2 * (
                self.gamma_hot_plus + self.gamma_cold_plus) ** 2 + self.gamma_hot_plus * self.gamma_cold_plus
              * self.Gamma ** 2) / self.chi

        r2 = (4 * self.g ** 2 * (self.gamma_hot_minus + self.gamma_cold_minus) * (
                self.gamma_hot_plus + self.gamma_cold_plus) + self.gamma_hot_plus * self.gamma_cold_minus
              * self.Gamma ** 2) / self.chi

        r3 = (4 * self.g ** 2 * (self.gamma_hot_minus + self.gamma_cold_minus) * (
                self.gamma_hot_plus + self.gamma_cold_plus) + self.gamma_hot_minus * self.gamma_cold_plus
              * self.Gamma ** 2) / self.chi

        r4 = (4 * self.g ** 2 * (
                self.gamma_hot_minus + self.gamma_cold_minus) ** 2 + self.gamma_hot_minus * self.gamma_cold_minus
              * self.Gamma ** 2) / self.chi

        c = 2j * self.g * self.Gamma * (
                self.gamma_hot_plus * self.gamma_cold_minus - self.gamma_hot_minus
                * self.gamma_cold_plus) / self.chi

        self.rho = qt.Qobj([
            [r1, 0, 0, 0],
            [0, r2, c, 0],
            [0, c.conjugate(), r3, 0],
            [0, 0, 0, r4]
        ], dims=[[2, 2], [2, 2]])
