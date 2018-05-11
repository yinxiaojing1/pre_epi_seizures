'''
Created on 14 de Jan de 2013

@author: Carlos
'''

import numpy as np
import pylab as pl
from itertools import cycle
from scipy import interpolate, signal
import matplotlib.font_manager as fm
# import matplotlib.patches as patches

# We need a special font for the code below.  It can be downloaded this way:
import os
import urllib2
if not os.path.exists('Humor-Sans.ttf'):
    fhandle = urllib2.urlopen('http://antiyawn.com/uploads/Humor-Sans.ttf')
    open('Humor-Sans.ttf', 'wb').write(fhandle.read())

def xkcd_line(x, y, xlim=None, ylim=None, mag=1.0, f1=30, f2=0.05, f3=15):
    """
    Mimic a hand-drawn line from (x, y) data

    Parameters
    ----------
    x, y : array_like
        arrays to be modified
    xlim, ylim : data range
        the assumed plot range for the modification.  If not specified,
        they will be guessed from the  data
    mag : float
        magnitude of distortions
    f1, f2, f3 : int, float, int
        filtering parameters.  f1 gives the size of the window, f2 gives
        the high-frequency cutoff, f3 gives the size of the filter
    
    Returns
    -------
    x, y : ndarrays
        The modified lines
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # get limits for rescaling
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    if xlim[1] == xlim[0]:
        xlim = ylim
        
    if ylim[1] == ylim[0]:
        ylim = xlim

    # scale the data
    x_scaled = (x - xlim[0]) * 1. / (xlim[1] - xlim[0])
    y_scaled = (y - ylim[0]) * 1. / (ylim[1] - ylim[0])

    # compute the total distance along the path
    dx = x_scaled[1:] - x_scaled[:-1]
    dy = y_scaled[1:] - y_scaled[:-1]
    dist_tot = np.sum(np.sqrt(dx * dx + dy * dy))

    # number of interpolated points is proportional to the distance
    Nu = int(200 * dist_tot)
    u = np.arange(-1, Nu + 1) * 1. / (Nu - 1)

    # interpolate curve at sampled points
    k = min(3, len(x) - 1)
    res = interpolate.splprep([x_scaled, y_scaled], s=0, k=k)
    x_int, y_int = interpolate.splev(u, res[0]) 

    # we'll perturb perpendicular to the drawn line
    dx = x_int[2:] - x_int[:-2]
    dy = y_int[2:] - y_int[:-2]
    dist = np.sqrt(dx * dx + dy * dy)

    # create a filtered perturbation
    coeffs = mag * np.random.normal(0, 0.01, len(x_int) - 2)
    b = signal.firwin(f1, f2 * dist_tot, window=('kaiser', f3))
    response = signal.lfilter(b, 1, coeffs)

    x_int[1:-1] += response * dy / dist
    y_int[1:-1] += response * dx / dist

    # un-scale data
    x_int = x_int[1:-1] * (xlim[1] - xlim[0]) + xlim[0]
    y_int = y_int[1:-1] * (ylim[1] - ylim[0]) + ylim[0]
    
    return x_int, y_int

def XKCDify(ax, mag=1.0, f1=50, f2=0.01, f3=15,
            bgcolor='w',
            xaxis_loc=None,
            yaxis_loc=None,
            xaxis_arrow='+',
            yaxis_arrow='+',
            ax_extend=0.1,
            expand_axes=False):
    """Make axis look hand-drawn

    This adjusts all lines, text, legends, and axes in the figure to look
    like xkcd plots.  Other plot elements are not modified.
    
    Parameters
    ----------
    ax : Axes instance
        the axes to be modified.
    mag : float
        the magnitude of the distortion
    f1, f2, f3 : int, float, int
        filtering parameters.  f1 gives the size of the window, f2 gives
        the high-frequency cutoff, f3 gives the size of the filter
    xaxis_loc, yaxis_log : float
        The locations to draw the x and y axes.  If not specified, they
        will be drawn from the bottom left of the plot
    xaxis_arrow, yaxis_arrow : str
        where to draw arrows on the x/y axes.  Options are '+', '-', '+-', or ''
    ax_extend : float
        How far (fractionally) to extend the drawn axes beyond the original
        axes limits
    expand_axes : bool
        if True, then expand axes to fill the figure (useful if there is only
        a single axes in the figure)
    """
    # Get axes aspect
    ext = ax.get_window_extent().extents
    aspect = (ext[3] - ext[1]) / (ext[2] - ext[0])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - xlim[0]

    xax_lim = (xlim[0] - ax_extend * xspan,
               xlim[1] + ax_extend * xspan)
    yax_lim = (ylim[0] - ax_extend * yspan,
               ylim[1] + ax_extend * yspan)

    if xaxis_loc is None:
        xaxis_loc = ylim[0]

    if yaxis_loc is None:
        yaxis_loc = xlim[0]

    # Draw axes
    xaxis = pl.Line2D([xax_lim[0], xax_lim[1]], [xaxis_loc, xaxis_loc],
                      linestyle='-', color='k')
    yaxis = pl.Line2D([yaxis_loc, yaxis_loc], [yax_lim[0], yax_lim[1]],
                      linestyle='-', color='k')

    # Label axes3, 0.5, 'hello', fontsize=14)
    ax.text(xax_lim[1], xaxis_loc - 0.02 * yspan, ax.get_xlabel(),
            fontsize=14, ha='right', va='top', rotation=12)
    ax.text(yaxis_loc - 0.02 * xspan, yax_lim[1], ax.get_ylabel(),
            fontsize=14, ha='right', va='top', rotation=78)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add title
    ax.text(0.5 * (xax_lim[1] + xax_lim[0]), yax_lim[1],
            ax.get_title(),
            ha='center', va='bottom', fontsize=16)
    ax.set_title('')

    Nlines = len(ax.lines)
    lines = [xaxis, yaxis] + [ax.lines.pop(0) for i in range(Nlines)]

    for line in lines:
        x, y = line.get_data()

        x_int, y_int = xkcd_line(x, y, xlim, ylim,
                                 mag, f1, f2, f3)

        # create foreground and background line
        lw = line.get_linewidth()
        line.set_linewidth(2 * lw)
        line.set_data(x_int, y_int)

        # don't add background line for axes
        if (line is not xaxis) and (line is not yaxis):
            line_bg = pl.Line2D(x_int, y_int, color=bgcolor,
                                linewidth=8 * lw)

            ax.add_line(line_bg)
        ax.add_line(line)

    # Draw arrow-heads at the end of axes lines
    arr1 = 0.03 * np.array([-1, 0, -1])
    arr2 = 0.02 * np.array([-1, 0, 1])

    arr1[::2] += np.random.normal(0, 0.005, 2)
    arr2[::2] += np.random.normal(0, 0.005, 2)

    x, y = xaxis.get_data()
    if '+' in str(xaxis_arrow):
        ax.plot(x[-1] + arr1 * xspan * aspect,
                y[-1] + arr2 * yspan,
                color='k', lw=2)
    if '-' in str(xaxis_arrow):
        ax.plot(x[0] - arr1 * xspan * aspect,
                y[0] - arr2 * yspan,
                color='k', lw=2)

    x, y = yaxis.get_data()
    if '+' in str(yaxis_arrow):
        ax.plot(x[-1] + arr2 * xspan * aspect,
                y[-1] + arr1 * yspan,
                color='k', lw=2)
    if '-' in str(yaxis_arrow):
        ax.plot(x[0] - arr2 * xspan * aspect,
                y[0] - arr1 * yspan,
                color='k', lw=2)

    # Change all the fonts to humor-sans.
    prop = fm.FontProperties(fname='Humor-Sans.ttf', size=16)
    for text in ax.texts:
        text.set_fontproperties(prop)
    
    # modify legend
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        
        for child in leg.get_children():
            if isinstance(child, pl.Line2D):
                x, y = child.get_data()
                child.set_data(xkcd_line(x, y, mag=10, f1=100, f2=0.001))
                child.set_linewidth(2 * child.get_linewidth())
            if isinstance(child, pl.Text):
                child.set_fontproperties(prop)
    
    # Set the axis limits
    ax.set_xlim(xax_lim[0] - 0.1 * xspan,
                xax_lim[1] + 0.1 * xspan)
    ax.set_ylim(yax_lim[0] - 0.1 * yspan,
                yax_lim[1] + 0.1 * yspan)

    # adjust the axes
    ax.set_xticks([])
    ax.set_yticks([])      

    if expand_axes:
        ax.figure.set_facecolor(bgcolor)
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
    
    return ax


def plotErr(waves, EER, IDErr):
    
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 16}
    pl.rc('font', **font)
    
    color1 = 'blue'
    color2 = 'yellow'
    
    width = 0.35
    ind = np.arange(len(waves))
    
    
    fig = pl.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)
    
    rects1 = ax1.bar(ind, EER, width, color=color1, hatch='')
    
    
    ax1.set_xticks(ind + width)
    ax1.set_xticklabels(waves)
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
    
    # ax1.legend((rects1[0],), ('EER (Autethentication)',), loc='best')
    ax1.set_ylabel('Authentication EER (%)')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(ind+width, IDErr, width, color=color2, hatch='//')
    
    ax2.set_xticks(ind + width)
    ax2.set_xticklabels(waves)
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
    
    ax2.legend((rects1[0], rects2[0]),
               ('Authentication EER', 'Identification Error',),
               loc='upper center',
               bbox_to_anchor=(0.75, 1.115),
               fancybox=True,
               shadow=True)
    ax2.set_ylabel('Identification Error (%)')
    
    return fig


def plotIDErr(waves, IDErr):
    
    fig = pl.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.plot(IDErr)
    
    return fig


def plotTests(waves, tests, labels=None, xlabel=None, ylabel=None, width=0.15, xlim=None, ylim=None, legendAnchor=(0.75, 1.115)):
    
    # font parameters
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'normal',
            'size': 16}
    pl.rc('font', **font)
    
    # fill colors and styles
    colors = cycle(['blue', 'yellow', 'cyan', 'green', 'red', 'gray', 'magenta'])
    hatches = cycle(['', '//', '\\', 'x', '++', 'o', 'O', '-', '||', '.', '*'])
    
    # make figure
    fig = pl.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # x indexes
    ind = np.arange(len(waves))
    nb = len(tests)
    
    # bars
    rects = []
    for i in xrange(nb):
        out = ax.bar(ind + i * width, tests[i], width, color=colors.next(), hatch=hatches.next())
        rects.append(out[0])
    
    # set xticks
    ax.set_xticks(ind + (nb / 2) * width)
    ax.set_xticklabels(waves)
    
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    
    # set xlim
    if xlim is None:
        ax.set_xlim(-2*width, ind[-1] + i * width + 3*width)
    else:
        ax.set_xlim(xlim[0], xlim[1])
    
    # set ylim
    if ylim is None:
        pass
    else:
        ax.set_ylim(ylim[0], ylim[1])
    
    # xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    # ylabel
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # legend
    if labels is not None:
        ax.legend(tuple(rects), labels,
                  loc='best', # 'upper center'
                  bbox_to_anchor=legendAnchor, # (0.75, 1.115)
                  fancybox=True,
                  shadow=True,
                  ncol=int(round(float(nb) / 2)))
    
    # grid
    ax.grid(which='major', axis='y', linewidth=1.5)
    
    return fig


def stats(tests):
    
    out = {}
    
    # global
    out['Global Mean'] = tests[:].mean()
    out['Global Std'] = tests[:].std(ddof=1)
    
    # over the lines
    out['Lines Mean'] = tests.mean(axis=0)
    out['Lines Std'] = tests.std(ddof=1, axis=0)
    
    # over the columns
    out['Columns Mean'] = tests.mean(axis=1)
    out['Columns Std'] = tests.std(ddof=1, axis=1)
    
    return out



if __name__ == '__main__':
    fpath = 'C:\\papers\\Wavelets_ICIAR2013\\Img\\%s.pdf'
    
    # results from enfermagem
    
    # wavelet names
    waves = ['rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio5.5', 'coif5', 'db3', 'db8']
    
    # results from test A1 (engzee, rcosdemean, T1/T2, median 5/5)
    EER_A1 = [11.8, 12.5, 13.4, 13.8, 11.6, 13.1, 11.5, 13.7]
    IDErr_A1 = [29.8, 28.6, 29.1, 29.1, 28.1, 26.0, 28.6, 34.0]
    # fig = plotErr(waves, EER_A, IDErr_A)
    
    
    # results from test B1 (wavelet, dbscan, T1/T2, median 5/5)
    EER_B1 = [12.5, 12.5, 13.5, 14.1, 12.0, 14.4, 14.1, 13.5]
    IDErr_B1 = [31.6, 29.5, 30.7, 32.5, 31.8, 31.8, 27.2, 29.2]
    # fig = plotErr(waves, EER_B, IDErr_B)
    
    # results from test C1 (engzee, dbscan, T1/T2, median 5/5)
    EER_C1 = [12.8, 12.7, 13.5, 14.5, 11.6, 13.7, 11.6, 13.7]
    IDErr_C1 = [30.3, 29.3, 29.3, 31.0, 27.5, 27.4, 30.3, 35.0]
    # fig = plotErr(waves, EER_C, IDErr_C)
    
    # results from test D1 (wavelet, rcosdmean_wavelet, T1/T2, median 5/5)
    EER_D1 = [12.7, 13.2, 13.7, 14.2, 11.9, 13.8, 14.4, 13.1]
    IDErr_D1 = [32.2, 31.3, 32.3, 33.9, 28.6, 31.4, 26.9, 29.1]
    # fig = plotErr(waves, EER_D, IDErr_D)
    
    # results from test E1 (engzee, rcosdmean_wavelet, T1/T2, median 5/5)
    EER_E1 = [13.3, 13.6, 14.2, 15.1, 12.8, 14.5, 12.0, 16.0]
    IDErr_E1 = [31.8, 31.7, 32.5, 32.5, 28.3, 29.0, 28.1, 37.4]
    
    
    labels_1 = ('Test A', 'Test B', 'Test C', 'Test D', 'Test E')
    EER_1 = np.array((EER_A1, EER_B1, EER_C1, EER_D1, EER_E1))
    IDErr_1 = np.array((IDErr_A1, IDErr_B1, IDErr_C1, IDErr_D1, IDErr_E1))
    
    #-------------------------------------------------------------------
    
    # results from test A2 (engzee, rcosdemean, T1/T2, mean 5/5)
    EER_A2 = [11.9, 12.5, 13.0, 13.6, 12.2, 13.0, 11.2, 11.9,]
    IDErr_A2 = [31.0, 29.9, 31.2, 31.3, 25.9, 27.8, 29.0, 30.9]
    # fig = plotErr(waves, EER_A, IDErr_A)
    
    
    # results from test B2 (wavelet, dbscan, T1/T2, mean 5/5)
    EER_B2 = [11.8, 12.9, 13.0, 13.5, 12.3, 14.1, 14.8, 11.3]
    IDErr_B2 = [32.5, 31.5, 32.4, 34.0, 30.9, 30.6, 26.5, 33.7]
    # fig = plotErr(waves, EER_B, IDErr_B)
    
    # results from test C2 (engzee, dbscan, T1/T2, mean 5/5)
    EER_C2 = [13.3, 13.3, 14.4, 14.8, 12.3, 14.3, 12.1, 13.5]
    IDErr_C2 = [32.0, 31.8, 30.3, 31.8, 27.3, 33.2, 29.1, 35.8]
    # fig = plotErr(waves, EER_C, IDErr_C)
    
    # results from test D2 (wavelet, rcosdmean_wavelet, T1/T2, mean 5/5)
    EER_D2 = [12.5, 13.0, 14.2, 14.3, 12.2, 13.5, 14.4, 12.7]
    IDErr_D2 = [33.2, 32.1, 34.5, 32.6, 27.9, 30.7, 26.4, 29.5]
    # fig = plotErr(waves, EER_D, IDErr_D)
    
    # results from test E2 (engzee, rcosdmean_wavelet, T1/T2, mean 5/5)
    EER_E2 = [13.2, 13.3, 14.0, 15.2, 14.1, 14.7, 12.5, 15.1]
    IDErr_E2 = [32.5, 33.9, 31.2, 33.6, 28.4, 32.0, 28.6, 36.3]
    
    
    labels_2 = ('Test A', 'Test B', 'Test C', 'Test D', 'Test E')
    EER_2 = np.array((EER_A2, EER_B2, EER_C2, EER_D2, EER_E2))
    IDErr_2 = np.array((IDErr_A2, IDErr_B2, IDErr_C2, IDErr_D2, IDErr_E2))
    
    #-------------------------------------------------------------------
    
    # results from entire population
    
    # wavelet names
    waves_all = ['rbio5.5', 'db3']
    
    # results from test A3 (engzee, rcosdemean, T1/T2, median 5/5)
    EER_A3 = [12.2, 12.3]
    IDErr_A3 = [36.1, 36.9]
    # fig = plotErr(waves, EER_A, IDErr_A)
    
    
    # results from test B3 (wavelet, dbscan, T1/T2, median 5/5)
    EER_B3 = [12.8, 12.8]
    IDErr_B3 = [36.0, 36.0]
    # fig = plotErr(waves, EER_B, IDErr_B)
    
    # results from test C3 (engzee, dbscan, T1/T2, median 5/5)
    EER_C3 = [11.9, 12.0]
    IDErr_C3 = [34.6, 36.1]
    # fig = plotErr(waves, EER_C, IDErr_C)
    
    # results from test D3 (wavelet, rcosdmean_wavelet, T1/T2, median 5/5)
    EER_D3 = [13.9, 14.1]
    IDErr_D3 = [36.6, 38.8]
    # fig = plotErr(waves, EER_D, IDErr_D)
    
    # results from test E3 (engzee, rcosdmean_wavelet, T1/T2, median 5/5)
    EER_E3 = [13.7, 13.2]
    IDErr_E3 = [35.5, 36.6]
    
    labels_3 = ('Test A', 'Test B', 'Test C', 'Test D', 'Test E')
    EER_3 = np.array((EER_A3, EER_B3, EER_C3, EER_D3, EER_E3))
    IDErr_3 = np.array((IDErr_A3, IDErr_B3, IDErr_C3, IDErr_D3, IDErr_E3))
    
    #-------------------------------------------------------------------
    
    # results from test A4 (engzee, rcosdemean, T1/T2, mean 5/5)
    EER_A4 = [12.8, 12.3]
    IDErr_A4 = [36.0, 37.5]
    # fig = plotErr(waves, EER_A, IDErr_A)
    
    
    # results from test B4 (wavelet, dbscan, T1/T2, mean 5/5)
    EER_B4 = [12.5, 12.6]
    IDErr_B4 = [35.4, 34.7]
    # fig = plotErr(waves, EER_B, IDErr_B)
    
    # results from test C4 (engzee, dbscan, T1/T2, mean 5/5)
    EER_C4 = [12.4, 12.3]
    IDErr_C4 = [35.1, 35.3]
    # fig = plotErr(waves, EER_C, IDErr_C)
    
    # results from test D4 (wavelet, rcosdmean_wavelet, T1/T2, mean 5/5)
    EER_D4 = [13.4, 14.7]
    IDErr_D4 = [36.2, 38.7]
    # fig = plotErr(waves, EER_D, IDErr_D)
    
    # results from test E4 (engzee, rcosdmean_wavelet, T1/T2, mean 5/5)
    EER_E4 = [14.1, 13.1]
    IDErr_E4 = [37.0, 36.7]
    
    labels_4 = ('Test A', 'Test B', 'Test C', 'Test D', 'Test E')
    EER_4 = np.array((EER_A4, EER_B4, EER_C4, EER_D4, EER_E4))
    IDErr_4 = np.array((IDErr_A4, IDErr_B4, IDErr_C4, IDErr_D4, IDErr_E4))
    
    #-------------------------------------------------------------------
    
    # authentication
    fig = plotTests(waves,
                    EER_1,
                    labels_1,
                    'EER (%)',
                    width=0.15,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_median', dpi=500)
    
    fig = plotTests(waves,
                    EER_2,
                    labels_2,
                    'EER (%)',
                    width=0.15,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_mean', dpi=500)
    
    fig = plotTests(waves_all,
                    EER_3,
                    labels_3,
                    'EER (%)',
                    width=0.15,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_median_all', dpi=500)
    
    fig = plotTests(waves_all,
                    EER_4,
                    labels_4,
                    'EER (%)',
                    width=0.15,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_mean_all', dpi=500)
    
    # identification
    fig = plotTests(waves,
                    IDErr_1,
                    labels_1,
                    'Identification Error (%)',
                    width=0.15,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_median', dpi=500)
    
    fig = plotTests(waves,
                    IDErr_2,
                    labels_2,
                    'Identification Error (%)',
                    width=0.15,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_mean', dpi=500)
    
    fig = plotTests(waves_all,
                    IDErr_3,
                    labels_3,
                    'Identification Error (%)',
                    width=0.15,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_median_all', dpi=500)
    
    fig = plotTests(waves_all,
                    IDErr_4,
                    labels_4,
                    'Identification Error (%)',
                    width=0.15,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_mean_all', dpi=500)
    
    # statistics
    EERS_1 = stats(EER_1)
#    print 'Authentication, Median'
#    print EERS_1
    
    EERS_2 = stats(EER_2)
#    print 'Authentication, Mean'
#    print EERS_2
    
    IDErrS_1 = stats(IDErr_1)
#    print 'Identification, Median'
#    print IDErrS_1
    
    IDErrS_2 = stats(IDErr_2)
#    print 'Identification, Mean'
#    print IDErrS_2
    
    fig = plotTests(waves,
                    (EERS_1['Lines Mean'], EERS_2['Lines Mean']),
                    ('Median', 'Mean'),
                    'EER (%)',
                    width=0.25,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_cmpWaves', dpi=500)
    
    fig = plotTests(waves,
                    (IDErrS_1['Lines Mean'], IDErrS_2['Lines Mean']),
                    ('Median', 'Mean'),
                    'Identification Error (%)',
                    width=0.25,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_cmpWaves', dpi=500)
    
    fig = plotTests(['Test A', 'Test B', 'Test C', 'Test D', 'Test E'],
                    (EERS_1['Columns Mean'], EERS_2['Columns Mean']),
                    ('Median', 'Mean'),
                    'EER (%)',
                    width=0.25,
                    ylim=(9, 17))
    fig.savefig(fpath % 'EER_cmpTests', dpi=500)
    
    fig = plotTests(['Test A', 'Test B', 'Test C', 'Test D', 'Test E'],
                    (IDErrS_1['Columns Mean'], IDErrS_2['Columns Mean']),
                    ('Median', 'Mean'),
                    'Identification Error (%)',
                    width=0.25,
                    ylim=(20, 40))
    fig.savefig(fpath % 'IDErr_cmpTests', dpi=500)
    
    pl.show()



    # test XKCDify
    
#    # ECG
#    import gzip
#    import cPickle
#    
#    fid = gzip.open('C:/Users/Carlos/Desktop/ecg.dict', 'rb')
#    data = cPickle.load(fid)
#    fid.close()
#    
#    fig = pl.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(range(len(data)), data, color='b', lw=1)
#    
#    ax.text(85, 95, 'P', color='r', ha='center')
#    ax.text(163, -135, 'Q', color='r', ha='center')
#    ax.text(200, 362, 'R', color='r', ha='center')
#    ax.text(257, -160, 'S', color='r', ha='center')
#    ax.text(394, 168, 'T', color='r', ha='center')
#    
#    XKCDify(ax, xaxis_loc=data.min()-20, yaxis_loc=-20, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)
#    
#    fig.savefig('C:/Users/Carlos/Desktop/ecg.png', bbox_inches='tight', dpi=500)
    
#    # EEG bands
#    fig = pl.figure()
#    ax = fig.add_subplot(111)
#    ax.plot([4, 8, 10, 13, 25, 40], [1, 1, 1, 1, 1, 1], 'b', lw=1)
#    ax.plot([0, 4], [0, 1], 'b', lw=1)
#    ax.plot([40, 45], [1, 0], 'b', lw=1)
#    ax.plot([4, 4], [0, 1], 'b', lw=1)
#    ax.plot([8, 8], [0, 1], 'b', lw=1)
#    ax.plot([10, 10], [0, 1], 'b', lw=1)
#    ax.plot([13, 13], [0, 1], 'b', lw=1)
#    ax.plot([25, 25], [0, 1], 'b', lw=1)
#    ax.plot([40, 40], [0, 1], 'b', lw=1)
#    
#    ax.text(4, -0.08, '4', color='r', ha='center')
#    ax.text(8, -0.08, '8', color='r', ha='center')
#    ax.text(10, -0.08, '10', color='r', ha='center')
#    ax.text(13, -0.08, '13', color='r', ha='center')
#    ax.text(25, -0.08, '25', color='r', ha='center')
#    ax.text(40, -0.08, '40', color='r', ha='center')
#    
#    ax.text(6.5, 1.12, 'Theta', color='k', ha='right')
#    ax.plot([6, 4.0], [1.02, 1.1], '-k', lw=0.5)
#    
#    ax.text(9, 1.28, 'Lower\nalpha', color='k', ha='left')
#    ax.plot([9, 9.5], [1.02, 1.26], '-k', lw=0.5)
#    
#    ax.text(11.5, 1.12, 'Upper\nalpha', color='k', ha='left')
#    ax.plot([11.5, 12], [1.02, 1.1], '-k', lw=0.5)
#    
#    ax.text(19, 1.12, 'Beta', color='k', ha='left')
#    ax.plot([19, 19.5], [1.02, 1.1], '-k', lw=0.5)
#    
#    ax.text(32.5, 1.12, 'Gamma', color='k', ha='left')
#    ax.plot([32, 33], [1.02, 1.1], '-k', lw=0.5)
#    
#    ax.set_xlabel('Frequency (Hz)')
#    ax.set_xlim(-0.2, 50)
#    ax.set_ylim(-0.2, 1.3)
#    
#    XKCDify(ax, xaxis_loc=0.0, yaxis_loc=0.0, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)
#    
#    fig.savefig('C:/Users/Carlos/Desktop/bands.png', bbox_inches='tight', dpi=500)
    
#    x = np.linspace(0, 10, 100)
#    
#    # XKDC
#    fig2 = pl.figure()
#    ax = fig2.add_subplot(111)
#    ax.plot(x, np.sin(x) * np.exp(-0.1 * (x - 5) ** 2), 'b', lw=1, label='damped sine')
#    ax.plot(x, -np.cos(x) * np.exp(-0.1 * (x - 5) ** 2), 'r', lw=1, label='damped cosine')
#    ax.set_title('check it out!')
#    ax.set_xlabel('x label')
#    ax.set_ylabel('y label')
#    ax.legend(loc='lower right')
#    ax.set_xlim(0, 10)
#    ax.set_ylim(-1.0, 1.0)
#    
#    #XKCDify the axes -- this operates in-place
#    XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)
    
    # pl.show()
    
