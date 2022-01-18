import numpy as np


def generate_plot_points(y_function, x_min, x_max, number_of_plot_points):
    step = (x_max - x_min) / number_of_plot_points
    x_range = np.arange(start=x_min, stop=x_max, step=step)
    y_range = []
    for x in x_range:
        y_range.append(y_function(x))
    return x_range, y_range
