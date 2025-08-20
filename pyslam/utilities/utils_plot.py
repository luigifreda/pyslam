"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt


# plot stats of scalar data
def plot_histogram(
    data,
    title="Histogram of Scalar Data",
    xlabel="Data",
    ylabel="Frequency",
    bins=30,
    show=True,
    is_positive_errors=False,
):

    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    standard_dev = np.sqrt(variance)

    bias = (
        median if not is_positive_errors else 0
    )  # We assume bias is 0 for positive errors. This is debeatable. Consider that in if errors are positive, the distribution is not even Gaussian and symmetric.
    # This is an heuristic that seems to work in practice

    absoulte_deviations = np.abs(data - bias)
    MAD = np.median(absoulte_deviations)
    sigma_mad = 1.4826 * MAD

    # print(f'Mean: {mean:.2f}, Median: {median:.2f}, Variance: {variance:.2f}, Standard Deviation: {standard_dev:.2f}, MAD: {MAD:.2f}, Sigma MAD: {sigma_mad:.2f}')
    # create a new figure
    plt.figure()

    # Plot histogram
    plt.hist(data, bins=bins, alpha=0.7, color="b", edgecolor="black")

    # Add lines for mean, median, and variance
    plt.axvline(mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2f}")
    plt.axvline(median, color="g", linestyle="dashed", linewidth=2, label=f"Median: {median:.2f}")

    lower_bound = mean - 3 * standard_dev if not is_positive_errors else 0
    upper_bound = mean + 3 * standard_dev
    if not is_positive_errors:
        plt.axvline(
            lower_bound,
            color="y",
            linestyle="dashed",
            linewidth=2,
            label=f"mean - 3*standard_dev: {lower_bound:.2f}",
        )
    plt.axvline(
        upper_bound,
        color="y",
        linestyle="dashed",
        linewidth=2,
        label=f"mean + 3*standard_dev: {upper_bound:.2f}",
    )

    # Add lines for MAD and sigma_mad
    lower_bound_mad = bias - 3 * sigma_mad if not is_positive_errors else 0
    upper_bound_mad = bias + 3 * sigma_mad
    if not is_positive_errors:
        plt.axvline(
            lower_bound_mad,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"bias - 3*sigma_mad: {lower_bound_mad:.2f}",
        )
    plt.axvline(
        upper_bound_mad,
        color="g",
        linestyle="dashed",
        linewidth=2,
        label=f"bias + 3*sigma_mad: {upper_bound_mad:.2f}",
    )

    # Display statistics on the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Show the plot
    if show:
        plt.show()


def plot_errors_histograms(
    data, title="Histogram of Scalar Data", xlabel="Data", ylabel="Frequency", bins=30, show=True
):
    return plot_histogram(data, title, xlabel, ylabel, bins, show, is_positive_errors=False)
