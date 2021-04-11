import autograd.numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# find corresponding value in Y at position x_opt
def value(x_opt, Y):
    range = int(len(Y)/200)
    pos = int((x_opt +range)/0.01)
    if pos < 0:
        pos = 0
    if pos >= len(Y) -1:
        pos = len(Y)-1
    return Y[pos]


# quantile line plot
def lineplot_qtl(x, y, n=20, percentile_min=1, percentile_max=99, color='r', label = 'line', plot_mean=True, plot_median=False, line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1/n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)


    if plot_mean:
        plt.plot(x, np.mean(y, axis=0), color=line_color, label = label)


    if plot_median:
        plt.plot(x, np.median(y, axis=0), color=line_color, label = label)
    
    return plt.gca()

















