import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def loglog_graph(nb_samples, MC_estims, ref_value):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, the estimators
           ref_value, the reference value
    return : /
    """
    # compute absolute values of the error between estimators and ref_value
    absolute_errors = np.abs(MC_estims - ref_value)

    # create log-log plot
    log_nb_samples = np.log10(nb_samples)
    log_errors = np.log10(absolute_errors)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_nb_samples, log_errors, label='Absolute error')

    # fit a linear regression line
    regression = LinearRegression()
    regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
    pred = regression.predict(log_nb_samples.reshape(-1, 1))
    plt.plot(log_nb_samples, pred, color='red', label='Linear regression')

    # get the coefficients of the linear regression
    slope = regression.coef_[0]
    intercept = regression.intercept_
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(1.5, -3, equation, fontsize=10, color='red')

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def plot_CI(nb_samples, estims, CI, exact_val, alpha):
    """
    Plot the confidence interval of the crude MC estimator.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, the estimators
           CI, confidence interval for the crude MC estimator based on these samples
           exact_val, the reference value
    return : /
    """
    plt.figure(figsize=(8, 6))
    plt.plot(nb_samples, estims, label='Estimator', color='red')
    plt.plot(nb_samples, CI[:, 0], linestyle='--', label='Estimated $ \\alpha $ = '+ str(alpha)+' confidence interval', color='orange')
    plt.plot(nb_samples, CI[:, 1], linestyle='--', color='orange')
    plt.axhline(y=exact_val, color='blue', linestyle='-.', label='True value')

    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Value')
    plt.title('Confidence interval for the estimator')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def loglog_average_error_graph(nb_samples, MC_estims, ref_value):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, list of lists of the estimators
           ref_value, the reference 'true' value
    return : /
    """
    # compute absolute values of the error between estimators and ref_value
    absolute_errors = np.abs(MC_estims - ref_value)
    
    averages = 1 / len(MC_estims[0]) * np.sum(absolute_errors, axis=1)

    # create log-log plot
    log_nb_samples = np.log10(nb_samples)
    log_errors = np.log10(averages)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_nb_samples, log_errors, label='Absolute error')

    # fit a linear regression line
    regression = LinearRegression()
    regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
    pred = regression.predict(log_nb_samples.reshape(-1, 1))
    plt.plot(log_nb_samples, pred, color='red', label='Linear regression')

    # get the coefficients of the linear regression
    slope = regression.coef_[0]
    intercept = regression.intercept_
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(2.8, -1.2, equation, fontsize=10, color='red')

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def multiple_loglog_graph(nb_samples, MC_estims_list, ref_value, legend_series):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, number of samples used to compute each estimator
           MC_estims_list, list of the estimators
           ref_value, the reference 'true' value
    return : /
    """    
    
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('Set2', len(MC_estims_list))  # Get distinct colors

    for i, MC_estims in enumerate(MC_estims_list):
        # discard values equal to -1 (not valid)
        valid_idx = np.where(MC_estims != -1)[0]
        MC_estims = MC_estims[valid_idx]
        
        # compute absolute values of the error between estimators and ref_value
        absolute_errors = np.abs(MC_estims - ref_value)

        # discard values for which the error is of the order of epsilon machine (cannot compute log)
        eps_idx = np.where(absolute_errors < 1e-16)[0]
        absolute_errors[eps_idx] = 1e-16

        # create log-log plot
        log_nb_samples = np.log10(nb_samples[valid_idx])
        log_errors = np.log10(absolute_errors)

        legend_M = legend_series[i]
        
        if legend_M == -2 or legend_M == -1 or legend_M == -3:
            
            if legend_M == -2:
                legend_M = 'n = M/2'
                color = 'blue'  # Blue color for legend_series -2
            elif legend_M == -1:
                legend_M = 'n = $\sqrt{M}$'
                color = 'red'  # Red color for legend_series -1
            elif legend_M == -3:
                legend_M = '$ M = n \log{n}$'
                color = 'green'
                
            plt.plot(log_nb_samples, log_errors, label=f'Serie ' + str(legend_M), color=color, linestyle='--', marker='o', markersize=3, linewidth=1)
            
        else:
            
            # colors for other legend_series values
            color = colors(i)

            plt.scatter(log_nb_samples, log_errors, label=f'Serie n = ' + str(legend_M), color=color, s=5)

            if len(log_nb_samples[log_nb_samples > 2.5]) != 0:
                # fit a linear regression line
                regression = LinearRegression()
                regression.fit(log_nb_samples[log_nb_samples > 2.5].reshape(-1, 1), log_errors[log_nb_samples > 2.5])
                pred = regression.predict(log_nb_samples[log_nb_samples > 2.5].reshape(-1, 1))
                plt.plot(log_nb_samples[log_nb_samples > 2.5], pred, color=color)

                slope = regression.coef_[0]
                intercept = regression.intercept_
                if intercept>=0:
                    equation = f'y = {slope:.2f}x + {intercept:.2f}'
                else:
                    equation = f'y = {slope:.2f}x - {np.abs(intercept):.2f}'
                plt.text(max(log_nb_samples) + 0.5, -1 - i * 0.5, equation, fontsize=10, color=color)

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def multiple_cond_loglog_graph(nb_samples, cond_list, legend_series):
    """
    Plot the graph of the condition number of the Vandermonde matrix.
    args : nb_samples, number of samples used to compute the least squares fit
           cond_list, list of the condition numbers
    return : /
    """    

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('Set2', len(cond_list))  # Get distinct colors

    for i, cond in enumerate(cond_list):
        color=colors(i)
        legend_M = "n =" + str(legend_series[i])
        if legend_series[i] == -1:
            color = 'red'
            legend_M = '$n = \sqrt{M}$'
        elif legend_series[i] == -2:
            color = 'blue'
            legend_M = 'n = M/2'
        elif legend_series[i] == -3:
            legend_M = '$ M = n \log{n}$'
            color = 'green'


        # discard values equal to -1 (not valid)
        valid_idx = np.where(cond != -1)[0]
        cond_filtered = cond[valid_idx]
        nb_samples_filtered = nb_samples[valid_idx]
        
        # create log-log plot
        log_nb_samples = np.log10(nb_samples_filtered)
        log_cond = np.log10(cond_filtered - 1)
    
        plt.plot(log_nb_samples, log_cond, label=f'Serie '+ legend_M, color=color, linestyle='--', marker='o', markersize=3, linewidth=1)

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Condition number - 1)')
    plt.title('Log-log plot of the condition number \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def visualize_cdf_from_samples(samples):
    """
    Plot a graph of empirical cdf, based on the received samples.
    args: samples, array containing the sample values
    return: /
    """
    M = len(samples)

    # sort the samples
    sorted_samples = np.sort(samples)

    # calculate the empirical CDF
    cdf = np.arange(1, M + 1) / M

    # plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_samples, cdf, label='Empirical CDF')
    plt.title('Empirical CDF')
    plt.xlabel('Samples')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()