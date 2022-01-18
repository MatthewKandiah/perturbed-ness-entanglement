import numpy as np
import qutip as qt
import pandas as pd
import matplotlib.pyplot as plt
import khandelwal_global_state

plot_hot_temperatures = np.arange(0.1, 60.1, 0.1)
plot_cold_temperatures = [0.01, 0.2, 0.25, 0.32, 0.4]
epsilon = 1
g = 0.02
gamma_cold = 0.1
gamma_hot = 0.001

result_array = []
plot_points_1 = [[], []]
plot_points_2 = [[], []]
plot_points_3 = [[], []]
plot_points_4 = [[], []]
plot_points_5 = [[], []]
for temperature_cold in plot_cold_temperatures:
    for temperature_hot in plot_hot_temperatures:
        parameter_dict = {
            'epsilon': epsilon,
            'g': g,
            'gamma_hot': gamma_hot,
            'gamma_cold': gamma_cold,
            'temperature_hot': temperature_hot,
            'temperature_cold': temperature_cold,
        }
        state = khandelwal_global_state.KhandelwalGlobalState(parameter_dict)
        concurrence = qt.concurrence(state.rho)
        result_array.append([epsilon, g, gamma_hot, gamma_cold, temperature_hot, temperature_cold, concurrence])

        if temperature_cold == plot_cold_temperatures[0]:
            plot_points_1[0].append(temperature_hot)
            plot_points_1[1].append(concurrence)
        elif temperature_cold == plot_cold_temperatures[1]:
            plot_points_2[0].append(temperature_hot)
            plot_points_2[1].append(concurrence)
        elif temperature_cold == plot_cold_temperatures[2]:
            plot_points_3[0].append(temperature_hot)
            plot_points_3[1].append(concurrence)
        elif temperature_cold == plot_cold_temperatures[3]:
            plot_points_4[0].append(temperature_hot)
            plot_points_4[1].append(concurrence)
        elif temperature_cold == plot_cold_temperatures[4]:
            plot_points_5[0].append(temperature_hot)
            plot_points_5[1].append(concurrence)

df = pd.DataFrame(data=result_array, columns=['epsilon', 'g', 'gamma_hot', 'gamma_cold', 'temperature_hot',
                                              'temperature_cold', 'concurrence'])
df.to_csv('khandelwal_global_concurrence.csv')

# plot concurrence against hot bath temperature at a range of cold bath temperatures
plt.plot(plot_points_1[0], plot_points_1[1])
plt.plot(plot_points_2[0], plot_points_2[1])
plt.plot(plot_points_3[0], plot_points_3[1])
plt.plot(plot_points_4[0], plot_points_4[1])
plt.plot(plot_points_5[0], plot_points_5[1])
plt.legend(plot_cold_temperatures)
plt.savefig('results.png')
