import numpy as np
import matplotlib.pyplot as plt





# plot stats of scalar data
def plot_histogram(data, title='Histogram of Scalar Data', xlabel='Data', ylabel='Frequency', bins=30, show=True, is_positive_errors=False):
    
    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    standard_dev = np.sqrt(variance)
    
    bias = median if not is_positive_errors else 0  # We assume bias is 0 for positive errors. This is debeatable. Consider that in if errors are positive, the distribution is not even Gaussian and symmetric. 
                                                    # This is an heuristic that seems to work in practice
    
    absoulte_deviations = np.abs(data-bias)
    MAD = np.median(absoulte_deviations)
    sigma_mad = 1.4826 * MAD
    
    #print(f'Mean: {mean:.2f}, Median: {median:.2f}, Variance: {variance:.2f}, Standard Deviation: {standard_dev:.2f}, MAD: {MAD:.2f}, Sigma MAD: {sigma_mad:.2f}')
    # create a new figure
    plt.figure()
      
    # Plot histogram
    plt.hist(data, bins=bins, alpha=0.7, color='b', edgecolor='black')

    # Add lines for mean, median, and variance
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    
    plt.axvline(mean - 3*np.sqrt(variance), color='y', linestyle='dashed', linewidth=2, label=f'mean-3*standard_dev: {mean - 3*standard_dev:.2f}')
    plt.axvline(mean + 3*np.sqrt(variance), color='y', linestyle='dashed', linewidth=2, label=f'mean+3*standard_dev: {mean + 3*standard_dev:.2f}')
    
    # Add lines for MAD and sigma_mad
    plt.axvline(bias - 3*sigma_mad, color='g', linestyle='dashed', linewidth=2, label=f'bias-3*sigma_mad: {bias - 3*sigma_mad:.2f}')
    plt.axvline(bias + 3*sigma_mad, color='g', linestyle='dashed', linewidth=2, label=f'bias+3*sigma_mad: {bias + 3*sigma_mad:.2f}')

    # Display statistics on the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Show the plot
    if show:
        plt.show()
        
        
def plot_errors_histograms(data, title='Histogram of Scalar Data', xlabel='Data', ylabel='Frequency', bins=30, show=True):
    return plot_histogram(data, title, xlabel, ylabel, bins, show, is_positive_errors=True)