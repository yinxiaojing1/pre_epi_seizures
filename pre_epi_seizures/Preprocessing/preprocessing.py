"""
Preprocessing pipeline
@author: Afonso Eduardo
"""
from __future__ import division

import os
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pre_epi_seizures.storage_utils
## Local imports
# Filtering functions
from Filtering import medianFIR, filterIR5to20, filter_signal
# R peak detectors
from Filtering import UNSW_RPeakDetector
# Smoothers
from Filtering import EKSmoothing, EKSmoothing17
# Oulier detection based on DMean
from outlier import dmean, wavedistance, cosdistance, msedistance

#==============================================================================
# IO Utility Functions
#==============================================================================
def read_dataset_csv(dfile, to_array=True, multicolumn=False, level=2, separate_instances=False):
    """
    Read a dataset from a csv where each column is a sample. Column labels are class labels.
    Use multicolumn if there are multiple samples per class. In this case, column labels are assumed to
    be multi-index where levels 0 and 1 are class label and sample number respectively. In addition, column
    labels are assumed to be ints and sorted.
    Parameters:
    -----------
    dfile: str
        Path of the csv file.
    to_array: bool (default: True)
        Cast dataframe to arrays X and y. If so, X is transposed, i.e. 1 sample per row.
    multicolumn: bool (default: False)
        If the csv file has multi-index columns.
    level: int (default: 2)
        Number of levels. Only applies when multicolumn.
    separate_instances: bool (default: False)
        If True and multicolumn is True, each instance label (y) is different up to the level as specified
        in the argument. In this particular case, y is an array of str (e.g. #1_..._#level).
    Returns:
    --------
    df: pd.DataFrame (to_array=False)
        Dataframe containing the dataset (1 sample per column).
    X: array 2D (to_array=True)
        Data (1 sample per row).
    y: array 1D (to_array=True)
        Labels (int or str if multicolumn and separate_instances).
    Notes:
    ------
    For futher information on multiindex, check pandas library.
    """
    df = pd.read_csv(dfile, index_col=0, header=range(level)) if multicolumn else pd.read_csv(dfile, index_col=0)
    if to_array:
        X = df.values.T
        if multicolumn and separate_instances:
            y = np.array(map(lambda x: '_'.join(map(lambda x: str(x), x)),
                         zip(*[df.columns.get_level_values(i).values.tolist() for i in range(level)])))
        else:
            y = np.array(map(int, df.columns.get_level_values(0) if multicolumn else df.columns))
        return X, y
    return df

def save_dataset_csv(X, y, dfile):
    """
    Saves a dataset into a csv file. If there are multiple samples per class (i.e. repeated y),
    create a multi-index column dataframe.
    Parameters:
    -----------
    X: array 2D
        Data (1 sample per row).
    y: array 1D
        Labels.
    dfile: str
        Path of the csv file.
    Notes:
    ------
    For futher information on multi-index, check pandas library.
    """
    uids = np.unique(y)
    if len(uids) != len(y): # if there are multiple samples per class, create multicolumn dataframe
        n_samples = [len(np.where(y==uid)[0]) for uid in uids]
        y =  pd.MultiIndex.from_arrays((y,
                                list(itertools.chain.from_iterable([range(n) for n in n_samples]))),
                                names=['label','sample'])
        if isinstance(X, list):
            X = np.vstack(X)
    pd.DataFrame(X.T, columns=y).to_csv(dfile)

read_R_csv = lambda dfile: [rpeaks[~np.isnan(rpeaks)].astype(int) for rpeaks in pd.read_csv(dfile, header=None).values]
save_R_csv = lambda R_list, dfile: pd.DataFrame(R_list).to_csv(dfile, header=False, index=False)
#==============================================================================

#==============================================================================
# Plotting Functions
#==============================================================================
def create_figure(data, savefolder, name, maximized=True):
    """
    Plot data by creating and saving a png figure.
    Parameters:
    -----------
    data: array 1D or 2D
        Data to plot (1 signal per column if 2D).
    savefolder: str
        Path to the save folder.
    name: str
        Name of the png image.
    maximized: bool (default: True)
        Whether to maximize window before saving.
    """
    plt.figure()
    plt.plot(data)
    if maximized:
        plt.get_current_fig_manager().window.showMaximized()
    plt.savefig(os.path.join(savefolder, '{}.png'.format(name)))
    plt.close()

def plot_dataset(dfile, savefolder=None, **kwargs):
    """
    Load dataset from a csv file and plot dataset by creating 1 image per class,
    i.e. samples from same class are plotted together.
    Parameters:
    -----------
    dfile: str
        Path of the csv file.
    savefolder: str (optional)
        If not provided, savefolder is the same as dfile folder.
    kwargs: dict
        Additional arguments to read_dataset_csv (to_array is True and cannot be changed).
    """
    if savefolder is None:
        savefolder = dfile.split('.')[0] + '_images'
    X, y = read_dataset_csv(dfile, to_array=True, **kwargs)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    plt.ioff()
    uids = np.unique(y)
    if len(uids) != len(y): # multiple samples per class
        for uid in uids:
            X_uid = np.atleast_2d(X[np.where(y==uid)[0]])
            create_figure(X_uid.T, savefolder, uid)
    else:
        for xx, yy in zip(X, y):
            create_figure(xx, savefolder, yy)

def plot_dataset_R(X, y, R_list, savefolder, mask_list=None):
    """Plot dataset with detected R peaks and artifact mask (optional)"""
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    plt.ioff()
    mask_list = [np.array([]),]*len(R_list) if mask_list is None else mask_list
    uids = np.unique(y)
    if len(uids) != len(y): # if multiple instances per class
        n_samples = [len(np.where(y==uid)[0]) for uid in uids]
        y = ['{}_{}'.format(uid, rec) for uid, rec in zip(y,
             list(itertools.chain.from_iterable([range(n) for n in n_samples])))]
    for x, yy, rpeaks, mask in zip(X, y, R_list, mask_list):
        plt.figure()
        plt.plot(x, 'k')
        if len(mask): plt.plot(mask, x[mask], 'r')
        if len(rpeaks): plt.plot(rpeaks, x[rpeaks], 'go')
        plt.get_current_fig_manager().window.showMaximized()
        plt.savefig(os.path.join(savefolder, '{}.png'.format(yy)))
        plt.close()

def plot_datasets(dfiles, subtitles=None, sharedylim=True, savefolder=None, **kwargs):
    """
    Plot several datasets on same figure, useful when comparing different
    filtering techniques. To align the datasets, their labels should match.
    When they don't, one (or more) subplots will appear blank.
    Parameters:
    -----------
    dfiles: list of str
        Paths of the csv files.
    subtitles: list of str
       Titles of each subplot. If None or len(subtitles)!=len(dfiles),
       the title is given the name of dfile.
    sharedylim: bool (default: True)
        If True, all subplots are set to same ylim.
    savefolder: str (optional)
        If not provided, savefolder is created in dfile1 folder, with the name
        being $dfile1name_vs_$dfile2name_vs_..._$dfileNname_images.
    kwargs: dict
        Additional arguments to read_dataset_csv (to_array is True and cannot be changed).
    """
    if savefolder is None:
        savefolder = '_vs_'.join([dfile.split('.')[0] if i==0 else os.path.split(dfile)[-1].split('.')[0] \
        for i, dfile in enumerate(dfiles)])
        savefolder +=  '_images'
    if subtitles is None or len(subtitles)!=len(dfiles):
        subtitles = map(lambda dfile: os.path.split(dfile)[-1].split('.')[0], dfiles)
    X_list, y_list = zip(*[read_dataset_csv(dfile, to_array=True, **kwargs) for dfile in dfiles])
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    plt.ioff()
    uids = np.unique(np.hstack(y_list))
    for uid in uids:
        X_list_uid = [np.atleast_2d(X[np.where(y==uid)[0]]) for X, y in zip(X_list, y_list)]
        f, ax_list = plt.subplots(len(dfiles),1)
        ymin, ymax = 1e10, -1e10
        for ax, X_uid, subtitle in zip(ax_list, X_list_uid, subtitles):
            if len(X_uid):
                ax.plot(X_uid.T)
                ax.set_title(subtitle)
                ymin, ymax = min(ymin, np.min(X_uid)), max(ymax, np.max(X_uid))
        if sharedylim:
            map(lambda ax:  ax.set_ylim([ymin, ymax]), ax_list)
        plt.get_current_fig_manager().window.showMaximized()
        plt.savefig(os.path.join(savefolder, '{}.png'.format(uid)))
        plt.close()
#==============================================================================


def create_filtered_dataset(dfile, filtmethod='medianFIR', save_dfile=None, multicolumn=False, **kwargs):
    """
    Load dataset from a csv file, filter with filtmethod save it.
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing 1 record per column.
    filtmethod: str (default: 'medianFIR')
        Name of the filter function defined in the global scope. For instance,
        'medianFIR' or 'filterIR5to20'.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, replace 'raw' with 'filtmethod'.
        If 'raw' does not exist in dfile name, append '_filtmethod'.
    multicolumn: bool (default: False)
        If the csv file to be loaded has multi-index columns. See
        read_dataset_csv and save_dataset_csv functions.
    kwargs: dict
        Additional arguments to filtmethod function (e.g. fs).
    """
    if save_dfile is None:
        save_dfile = dfile.replace('raw', filtmethod) if 'raw' in dfile else dfile.split('.')[0] + '_{}.csv'.format(filtmethod)
    X, y = read_dataset_csv(dfile, to_array=True, multicolumn=multicolumn) # 1 record per row
    X_filt = globals()[filtmethod](X, **kwargs)
    save_dataset_csv(X_filt, y, save_dfile)

def dataset_segmentation(dfile, save_dfile=None, rpeaks_list=None, rdetector='UNSW_RPeakDetector', lim=[-100, 250], **kwargs):
    """
    Load dataset from csv file, apply ECG segmentation using fixed-sized window around detected R peaks,
    save the resulting segments into a csv (multi-index dataframe).
    The R peaks are detected using rdetector function.
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing 1 record per column.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, append '_segments' to dfile name.
    rpeaks_list: list of arrays (default: None)
        R peak locations (1 array per signal). If None, computes the rpeaks using rdetector.
    rdetector: str (default: UNSW_RPeakDetector)
        Name of the R peak detector function defined in the global scope. For instance,
        'UNSW_RPeakDetector'.
    lim: list of int (default: [-100, 250])
        Lower and upper bound w.r.t detected R peaks [number of samples].
    kwargs: dict
        Additional arguments to rdetector function (e.g. fs).
    """
    X, y = read_dataset_csv(dfile, to_array=True) # 1 record per row
    rpeaks_list = globals()[rdetector](X, **kwargs) if rpeaks_list is None else rpeaks_list
    X_new, y_new = [], []
    lb, ub = lim
    ssize = ub-lb
    for xx, yy, Rpeaks in zip(X, y, rpeaks_list):
        if len(Rpeaks)==0:
            continue
        xx_segments = np.vstack([xx[lb+rpeak:ub+rpeak] for rpeak in Rpeaks if len(xx[lb+rpeak:ub+rpeak])==ssize])
        y_new.append([yy]*len(xx_segments))
        X_new.append(xx_segments)
    y_new = np.array(list(itertools.chain.from_iterable(y_new)))
    if len(y_new) == 0:
        raise ValueError('Could not perform segmentation on any record.')
    if save_dfile is None:
        save_dfile = dfile.split('.')[0] + '_segments.csv'
    save_dataset_csv(X_new, y_new, save_dfile)

def outlier_removal(dfile, save_dfile=None, min_samples=5, enable_dmean=False,
                    R_position=100, alpha=1.0, beta=1.5, metric='euclidean', checkR=False,
                    checkaroundR=2): # multiple samples per class database
    """
    Remove records/templates that have been deemed as outliers. If dmean operates at
    template level. In addition, there is the checkaroundR check and meanwave thresholding.
    Regarding meanwave, from Section 3.1 of [1]:  records "whose mean distances to the
    respective mean wave template are over the upper fence, i.e. Q3 + 1.5*(Q3-Q1),
    where Q1 and Q3 are the first and third quartiles, are classified as outliers and discarded".
    Parameters:
    -----------
    dfile: str
        Path pointing to csv file containing segments (templates), multi-index dataframe.
    save_dfile: str (default: None)
        Path pointing to new csv file. If None, append '_wo_out' to dfile name.
    min_samples: int (default: 5)
        Number of minimum samples a record must have to not be discarded.
    R_position: int (default: 100)
        Position of R peaks [index].
    checkaroundR: int (default: 2)
        If an integer is provided, check if the argmax of the signal is in the neighborhood
        of R_position such that |argmax - R_position| < checkaroundR. If any sample does not
        satisfy this condition, the record is discarded. If None, this is disabled.
    enable_dmean: bool (default: False)
        Whether to use dmean.
    alpha: float (default: 1.0)
        Dmean control parameter.
    beta: float (default: 1.5)
        Dmean control parameter.
    checkR: bool (default: False)
        Dmean control parameter.
    Notes:
    ------
    A. Eduardo, ECG-based Biometrics using a Deep Autoencoder for Feature Learning, 2016.
    """
    X, y = read_dataset_csv(dfile, to_array=True, multicolumn=True)
    uids = np.unique(y)
    df_wv = pd.DataFrame(columns=['uid', 'wvdist'])
    X_new, y_new = [], []
    for uid in uids:
        X_uid = np.atleast_2d(X[np.where(y==uid)[0]])
        if checkaroundR is not None and np.any(np.abs(np.argmax(X_uid, axis=1) - R_position) > checkaroundR):
            continue
        # remove samples based on dmean
        if enable_dmean:
            use_idx = np.array(dmean(data=X_uid, R_Position=R_position, metric=metric, alpha=alpha, beta=beta, checkR=checkR)['0'])
        else:
            use_idx = np.arange(len(X_uid))
        if len(use_idx) < min_samples:
            continue
        X_new.append(X_uid[use_idx])
        y_new.append([uid]*len(use_idx))
        # compute mean wave distance to meanwave
        meanwave = np.mean(X_new[-1], axis=0)
        wvdist = np.mean(wavedistance(meanwave, X_new[-1], msedistance))
        df_wv.loc[len(df_wv)] = [uid, wvdist]
    X_new, y_new = np.vstack(X_new), np.array(list(itertools.chain.from_iterable(y_new)))
    # remove users based on wave distance
    stats = df_wv['wvdist'].describe()
    wvdist_thre = stats['75%']+1.5*(stats['75%'] - stats['25%'])
    keep_idx = np.arange(len(y_new))[np.in1d(y_new, df_wv[df_wv['wvdist'] < wvdist_thre]['uid'].values)]
    X_new, y_new = X_new[keep_idx], y_new[keep_idx]
    if save_dfile is None:
        save_dfile = dfile.split('.')[0] + '_wo_out.csv'
    save_dataset_csv(X_new, y_new, save_dfile)


if __name__ == '__main__':
    f_path = lambda datafile: os.path.abspath("..\{}\{}".format(datafolder, datafile))

    datafolder = 'Data'
    fs = 500. # sampling frequency [Hz]

#################################################################################################################
# UNSW APPROACH - ECGIDDB_raw.csv (multiple sessions per subject)
#################################################################################################################
    step1, step2 = False, True
    if step1:
        X, y = read_dataset_csv(f_path("ECGIDDB_raw.csv"), multicolumn=True)
        R_list, mask_list = UNSW_RPeakDetector(X, fs=fs, artifact_masking=True, railV=[-10, 10], return_mask=True,
                                              padding=True, lp_b4_padding=True,
                                              check_peak=True, neighbors=20,
                                              check_amplitude=True, amplitude_thre=-.4, amplitude_thre_edges=-.4)
        X = medianFIR(X, fs=fs, lpfilter=True)
        plot_dataset_R(X, y, R_list, savefolder=f_path("ECGIDDB_medianFIR_UNSW_images"), mask_list=mask_list)
        keep_idx = np.array([i for i, mask in enumerate(mask_list) if len(mask)==0], dtype=int) # discard records based on artifact mask
        X, y, R_list = X[keep_idx], y[keep_idx], [R_list[i] for i in keep_idx]
        save_dataset_csv(X, y, dfile=f_path("ECGIDDB_medianFIR_UNSW.csv"))
        save_R_csv(R_list, dfile=f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
    if step2:
        R_list = read_R_csv(f_path("ECGIDDB_raw_UNSW_Rpeaks.csv"))
        #create_filtered_dataset(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True, filtmethod='EKSmoothing',
        #                        R_list=R_list, fs=fs, bins=250, verbose=True, oset=False, savefolder=f_path('ECGIDDB_EKSmoothing_images'))
        create_filtered_dataset(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True, filtmethod='EKSmoothing17',
                                R_list=R_list, fs=fs, bins=250, verbose=True, savefolder=f_path('ECGIDDB_EKSmoothing17_images'))        
        plot_datasets([f_path("ECGIDDB_medianFIR_UNSW.csv"), f_path("ECGIDDB_medianFIR_UNSW_EKSmoothing.csv"), 
                       f_path("ECGIDDB_medianFIR_UNSW_EKSmoothing17.csv")], multicolumn=True, separate_instances=True)
        


        #X, y = read_dataset_csv(f_path("ECGIDDB_medianFIR_UNSW.csv"), multicolumn=True)
        #x = X[1]
        #r = R_list[1]
        #yy = y[1]
        # plt.figure("Record {}".format(yy))
        # ax1 = plt.subplot(3, 2, 1)
        # ax1.set_title("medianFIR, full record")
        # ax1.plot(x)
        # ax1.plot(r, x[r], 'ro')
        
        #xeks = EKSmoothing(x, [r], fs=500., bins=250, verbose=True, oset=False)[0]
        # ax2 = plt.subplot(3, 2, 2, sharex=ax1)
        # ax2.set_title("medianFIR+EKS, full record")
        # ax2.plot(xeks)
        # ax2.plot(r, xeks[r], 'ro')

        # x_half1 = x[:int(len(x)/2)]
        # r_half1 = r[r < int(len(x)/2)]
        # ax3 = plt.subplot(3, 2, 3, sharex=ax1)
        # ax3.set_title("medianFIR, 1st half record")
        # ax3.plot(x_half1)
        # ax3.plot(r_half1, x_half1[r_half1], 'ro')

        # xeks_half1 = EKSmoothing(x_half1, [r_half1], fs=500., bins=250, verbose=True)[0]
        # ax4 = plt.subplot(3,2,4, sharex=ax1)
        # ax4.set_title("medianFIR+EKS, 1st half record")
        # ax4.plot(xeks_half1)
        # ax4.plot(r_half1, xeks_half1[r_half1], 'ro')

        # x_half2 = x[int(len(x)/2):]
        # r_half2 = r[r >= int(len(x)/2)] - int(len(x)/2) 
        # idx_half2 = np.arange(int(len(x)/2), len(x))
        # ax5 = plt.subplot(3,2,5, sharex=ax1)
        # ax5.set_title("medianFIR, 2nd half record")
        # ax5.plot(idx_half2, x_half2)
        # ax5.plot(r_half2 + int(len(x)/2), x_half2[r_half2], 'ro')

        # xeks_half2 = EKSmoothing(x_half2, [r_half2], fs=500., bins=250, verbose=True)[0]
        # ax6 = plt.subplot(3,2,6, sharex=ax1)
        # ax6.set_title("medianFIR+EKS, 2nd half record")
        # ax6.plot(idx_half2, xeks_half2)
        # ax6.plot(r_half2 + int(len(x)/2), xeks_half2[r_half2], 'ro')

        # plt.show()



################################################################################################################


#################################################################################################################
# UNSW APPROACH - healthy_raw.csv (one session per subject)
#################################################################################################################
    step1, step2, step3, step4 = False, False, False, False
    if step1:
        ## STEP 1 - create dataset without records that were deemed of poor quality or have features that deviate
        ##          from the idealized ECG model (e.g. waves other than R with higher amplitude). UNSW_RPeakDetector
        ##          (with additional modifications) is expected to identify all true R peaks: healthy_raw_UNSW.csv,
        ##          healthy_raw_UNSW_Rpeaks.csv.
        X, y = read_dataset_csv(f_path("healthy_raw.csv"))
        remove_ids = np.array([np.where(y==uid)[0][0] for uid in [1391, 1761, 1816, 2018]]) # waves other than R with higher amplitude
        X, y = np.delete(X, remove_ids,axis=0), np.delete(y, remove_ids)
        R_list, mask_list = UNSW_RPeakDetector(X, fs=fs, artifact_masking=True, return_mask=True, check_peak=True,
                                               padding=True, check_amplitude=True)
        keep_idx = np.array([i for i, mask in enumerate(mask_list) if len(mask)==0], dtype=int) # discard records based on artifact mask
        X, y, R_list = X[keep_idx], y[keep_idx], [R_list[i] for i in keep_idx]
        save_dataset_csv(X, y, dfile=f_path("healthy_raw_UNSW.csv"))
        save_R_csv(R_list, dfile=f_path("healthy_raw_UNSW_Rpeaks.csv"))
        plot_dataset_R(X, y, R_list, savefolder=f_path("healthy_raw_UNSW_images")) # plot raw dataset
        # remove baseline wander from raw dataset and plot the resulting dataset
        X = medianFIR(X, fs=fs, lpfilter=False)
        save_dataset_csv(X, y, dfile=f_path("healthy_median_UNSW.csv"))
        plot_dataset_R(X, y, R_list, savefolder=f_path("healthy_median_UNSW_images"))
    if step2:
        ## STEP 2 - create filtered dataset (baseline removal and lowpass of 40Hz): healthy_medianFIR_UNSW.csv
        R_list = read_R_csv(f_path("healthy_raw_UNSW_Rpeaks.csv"))
        X, y = read_dataset_csv(f_path("healthy_raw_UNSW.csv"))
        X = medianFIR(X, fs=fs, lpfilter=True)
        save_dataset_csv(X, y, dfile=f_path("healthy_medianFIR_UNSW.csv"))
        plot_dataset_R(X, y, R_list, savefolder=f_path("healthy_medianFIR_UNSW_images"))
        plot_datasets(map(lambda x: f_path(x),["healthy_raw_UNSW.csv", "healthy_median_UNSW.csv", "healthy_medianFIR_UNSW.csv"]))
    if step3:
        ## STEP 3 - create smoothed datasets and plot them for comparison: healthy_median_UNSW_EKSmoothing.csv
        ##          and healthy_medianFIR_UNSW_EKSmoothing.csv
        R_list = read_R_csv(f_path("healthy_raw_UNSW_Rpeaks.csv"))
        create_filtered_dataset(f_path("healthy_medianFIR_UNSW.csv"), filtmethod='EKSmoothing', R_list=R_list, fs=fs, bins=250, verbose=True)
        plot_datasets(map(lambda x: f_path(x),["healthy_medianFIR_UNSW.csv", "healthy_medianFIR_UNSW_EKSmoothing.csv"]))
        create_filtered_dataset(f_path("healthy_median_UNSW.csv"), filtmethod='EKSmoothing', R_list=R_list, fs=fs, bins=250, verbose=True)
        plot_datasets(map(lambda x: f_path(x),["healthy_median_UNSW.csv", "healthy_median_UNSW_EKSmoothing.csv", "healthy_medianFIR_UNSW_EKSmoothing.csv"]))
    if step4:
        ## STEP 4 - create datasets using other filtering approaches for future performance comparison; perform dataset segmentation
        create_filtered_dataset(f_path("healthy_raw_UNSW.csv"), filtmethod='filterIR5to20', fs=fs)
        R_list = read_R_csv(f_path("healthy_raw_UNSW_Rpeaks.csv"))
        for dfile in map(lambda x: f_path(x), ["healthy_filterIR5to20_UNSW.csv", "healthy_median_UNSW_EKSmoothing.csv",
                                               "healthy_medianFIR_UNSW_EKSmoothing.csv", "healthy_median_UNSW.csv",
                                               "healthy_medianFIR_UNSW.csv"]):
            dataset_segmentation(dfile, rpeaks_list=R_list, lim=[-100, 250])
################################################################################################################



#################################################################################################################
# OLD APPROACH - healthy_raw.csv (one session per subject)
#################################################################################################################
#    ## STEP 1 - create filtered dataset using medianFIR: healthy_medianFIR.csv
#    create_filtered_dataset(dfile=f_path('healthy_raw.csv'), filtmethod='medianFIR', fs=fs)
#
#    ## STEP 2 - create segmented dataset from filtered medianFIR dataset: healthy_medianFIR_segments.csv
#    dfile = f_path('healthy_medianFIR.csv')
#    plot_dataset(dfile)
#    dataset_segmentation(dfile, fs=fs)
#
#    ## STEP 3 - remove "outliers" from the segmented dataset: healthy_medianFIR_segments_wo_out.csv
#    dfile = f_path('healthy_medianFIR_segments.csv')
#    plot_dataset(dfile, multicolumn=True)
#    outlier_removal(dfile)
#
#    ## STEP 4 - fetch the corresponding records: healthy_medianFIR_wo_out.csv
#    dfile = f_path('healthy_medianFIR_segments_wo_out.csv')
#    plot_dataset(dfile, multicolumn=True)
#    # original without outliers
#    uids = np.unique(read_dataset_csv(dfile, multicolumn=True)[1])
#    orig_dfile = f_path('healthy_medianFIR.csv')
#    dset = read_dataset_csv(orig_dfile, to_array=False)[map(str, uids)]
#    dset.to_csv(f_path('healthy_medianFIR_wo_out.csv'))
#    plot_dataset(f_path('healthy_medianFIR_wo_out.csv'))
#
#    ## STEP 5 - from healthy_raw.csv pick the selected classes from the previous step: healthy_raw_selected.csv
#    X, y = read_dataset_csv(f_path('healthy_medianFIR_wo_out.csv'))
#    classes = np.unique(y)
#    X, y = read_dataset_csv(f_path('healthy_raw.csv'))
#    keep_idx = np.in1d(y, classes)
#    X, y = X[keep_idx], y[keep_idx]
#    save_dataset_csv(X, y, f_path('healthy_selected_raw.csv'))
#
#    ## STEP 6 - create filtered dataset using 5to20Hz filterIR and run segmentation
#    create_filtered_dataset(dfile=f_path('healthy_selected_raw.csv'), filtmethod='filterIR5to20', fs=fs)
#    dataset_segmentation(f_path('healthy_selected_filterIR5to20.csv'), fs=fs)
#    plot_dataset(f_path('healthy_selected_filterIR5to20_segments.csv'), multicolumn=True)
################################################################################################################
