import qutip as qt
import state_perturbation_lib as spl


class PottsState:

    def __init__(self, parameter_dict):
        self.kappa_prime_cold = parameter_dict['kappa_prime_cold']
        self.kappa_prime_hot = parameter_dict['kappa_prime_hot']
        self.epsilon = parameter_dict['epsilon']
        self.g = parameter_dict['g']
        self.temperature_cold = parameter_dict['temperature_cold']
        self.temperature_hot = parameter_dict['temperature_hot']
        self.beta_cold = 1. / self.temperature_cold
        self.beta_hot = 1. / self.temperature_hot

        self.hot_qubit_hamiltonian = qt.tensor(qt.qeye(2), qt.sigmaz()) * self.epsilon / 2
        self.cold_qubit_hamiltonian = qt.tensor(qt.sigmaz(), qt.qeye(2)) * self.epsilon / 2

        ket_e = qt.basis(2, 0)
        ket_g = qt.basis(2, 1)
        bra_e = ket_e.dag()
        bra_g = ket_g.dag()
        projector_ee = ket_e * bra_e
        projector_gg = ket_g * bra_g
        projector_ee_cold = qt.tensor(projector_ee, qt.qeye(2))
        projector_gg_cold = qt.tensor(projector_gg, qt.qeye(2))
        projector_ee_hot = qt.tensor(qt.qeye(2), projector_ee)
        projector_gg_hot = qt.tensor(qt.qeye(2), projector_gg)

        tau_cold = spl.nfd(self.epsilon, self.beta_cold) * projector_ee_cold + (
                    1 - spl.nfd(self.epsilon, self.beta_cold) * projector_gg_cold)
        tau_hot = spl.nfd(self.epsilon, self.beta_cold) * projector_ee_hot + (
                    1 - spl.nfd(self.epsilon, self.beta_cold) * projector_gg_hot)

        self.kappa_hot = (self.kappa_prime_hot * spl.nbe(self.epsilon, self.beta_hot)
                          / spl.nfd(self.epsilon, self.beta_hot))
        self.kappa_cold = (self.kappa_prime_cold * spl.nbe(self.epsilon, self.beta_cold)
                           / spl.nfd(self.epsilon, self.beta_cold))

        n_bar = (self.kappa_cold * spl.nfd(self.epsilon, self.beta_cold) + self.kappa_hot * spl.nfd(self.epsilon,
                                                                                                    self.beta_hot)) / (
                            self.kappa_cold + self.kappa_hot)
        tau_bar_cold = n_bar * projector_ee_cold + (1 - n_bar) * projector_gg_cold
        tau_bar_hot = n_bar * projector_ee_hot + (1 - n_bar) * projector_gg_hot

        sig_plus_cold = qt.tensor(qt.sigmap(), qt.qeye(2))
        sig_minus_cold = qt.tensor(qt.sigmam(), qt.qeye(2))
        sig_plus_hot = qt.tensor(qt.qeye(2), qt.sigmap())
        sig_minus_hot = qt.tensor(qt.qeye(2), qt.sigmam())

        self.rho = (
                (self.kappa_cold * self.kappa_hot * tau_cold * tau_hot) / (
                    self.kappa_cold * self.kappa_hot + 4 * (self.g ** 2))
                + (4 * (self.g ** 2) * tau_bar_cold * tau_bar_hot) / (
                            self.kappa_cold * self.kappa_hot + 4 * (self.g ** 2))
                + (2j * self.g * self.kappa_cold * self.kappa_hot * (
                    spl.nfd(self.epsilon, self.beta_hot) - spl.nfd(self.epsilon, self.beta_cold)) * (
                               sig_plus_cold * sig_minus_hot - sig_plus_hot * sig_minus_cold)) / (
                            (self.kappa_cold + self.kappa_hot) * (self.kappa_cold * self.kappa_hot + 4 * (self.g ** 2)))
        )
