import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.stats import kurtosis, skew



def lrt_compare(nu_val, yi, yi_1):
    res = yi

    if yi_1 <= yi <= (1 + nu_val) * yi_1:
        res = yi

    elif (yi < yi_1) or (yi > (1 + nu_val) * yi_1):
        res = yi_1 + nu_val

    return res


# function to apply LRT  to the data in window
def lrt_apply(data_window, prev_window_sample):
    data_window_rec = data_window.copy()

    # calculate the nu value over the data window of interest
    nu_sum = data_window_rec.diff()
    nu_val = nu_sum.mean()

    for i in range(len(data_window)):
        # handle the first window value index comparison
        if i == 0:
            data_window_rec.iat[i] = lrt_compare(nu_val, data_window[i], prev_window_sample)

        # perform the rectification for the rest of the index values
        else:
            data_window_rec.iat[i] = lrt_compare(nu_val, data_window[i], data_window[i - 1])

    # return updated window
    return data_window_rec


def plot_rms_vals(rms_df, columnName='rmsB1'):
    # individual RMS value
    fig = plt.figure(figsize=(10, 5))
    plt.plot(rms_df.date_time_format[:-2], rms_df[columnName][:-2], label= columnName, c="maroon")
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel(columnName)
    plt.grid(True)
    plt.show(block=False)
    figpath = "Results_Folder/" + columnName + '.png'
    fig.savefig(figpath)
    plt.pause(3)
    plt.close()


def prepare_data():
    listpath = os.walk(os.getcwd() + "/2nd_test/")
    all_path = []
    for i in listpath:
        all_path.append(i)

    rms_list_all_events = []

    columns = ["B1", "B2", "B3", "B4"]

    for i in range(len(all_path[0][2])):
        if all_path[0][2][i] == ".DS_Store":
            print("ignore string")
        else:
            path = all_path[0][0] + all_path[0][2][i]
            df = pd.read_csv(path, sep='\t', names=columns)

            rmsB1 = np.sqrt(np.mean(df.B1 ** 2))
            rmsB2 = np.sqrt(np.mean(df.B2 ** 2))
            rmsB3 = np.sqrt(np.mean(df.B3 ** 2))
            rmsB4 = np.sqrt(np.mean(df.B4 ** 2))

            kurtosis1 = kurtosis(df.B1)
            kurtosis2 = kurtosis(df.B2)
            kurtosis3 = kurtosis(df.B3)
            kurtosis4 = kurtosis(df.B4)

            stdn1 = np.sqrt(np.sum((np.array(df.B1) - np.mean(df.B1)) ** 2) / (len(df.B1) - 1))
            stdn2 = np.sqrt(np.sum((np.array(df.B2) - np.mean(df.B2)) ** 2) / (len(df.B2) - 1))
            stdn3 = np.sqrt(np.sum((np.array(df.B3) - np.mean(df.B3)) ** 2) / (len(df.B3) - 1))
            stdn4 = np.sqrt(np.sum((np.array(df.B4) - np.mean(df.B4)) ** 2) / (len(df.B4) - 1))

            mean1 = np.mean(df.B1)
            mean2 = np.mean(df.B2)
            mean3 = np.mean(df.B3)
            mean4 = np.mean(df.B4)

            skew1 = skew(df.B1)
            skew2 = skew(df.B2)
            skew3 = skew(df.B3)
            skew4 = skew(df.B4)

            dateString = all_path[0][2][i]
            str_dt = dateString[:10].replace(".", "-") + " " + dateString[11:].replace(".", ":")

            datetime_object = datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S')

            timestamp = int(time.mktime(datetime_object.timetuple()))

            data = {"rmsB1": rmsB1, "rmsB2":rmsB2, "rmsB3":rmsB3, "rmsB4":rmsB4,
                    "time": datetime_object.time(), "date_time_format" : datetime_object, "day":datetime_object.date().day,
                    "hour":datetime_object.time().hour, "date_time": timestamp,
                    "date":datetime_object.date(),"kurtosis1": kurtosis1, "kurtosis2": kurtosis2,
                    "kurtosis3":kurtosis3, "kurtosis4":kurtosis4,"skew1":skew1, "skew2":skew2,
                     "skew3":skew3, "skew4":skew4, "stdn1":stdn1, "stdn2": stdn2, "stdn3":stdn3,
                    "stdn4":stdn4, "mean1":mean1, "mean2":mean2, "mean3":mean3, "mean4":mean4,
                    "year":datetime_object.date().year,"month":datetime_object.date().month,
                    "day":datetime_object.date().day,"hour":datetime_object.time().hour,
                     "minute":datetime_object.time().minute,"second":datetime_object.time().second}

            rms_list_all_events.append(data)

    rms_df = pd.DataFrame(rms_list_all_events)

    # sort the data by date time
    rms_df.sort_values(["date_time"], axis=0, ascending=[True], inplace=True)

    # reset the index after sort operation
    rms_df = rms_df.reset_index(drop=True)

    return rms_df


def rolling_window():
    window_size = 50

    # initialization of the first i-1 element
    prev_window_val = 0

    time_sample = range(0, len(rms_df.rmsB1), window_size)

    # create a column to hold LRT applied values
    rms_df['rmsB1_LRT'] = 0

    # start binning the entire rms values based on windows size of 50
    for i in range(len(time_sample)):
        # handle the last bin
        if i == len(time_sample) - 1:
            start = time_sample[i]
            end = len(rms_df.rmsB1)
            print('Printing the last bin...')
            print(rms_df.rmsB1[start:end])

            rmsB1_slice = rms_df.rmsB1[start:end].copy().reset_index(drop=True)

            rms_df.rmsB1_LRT.iloc[start:end] = np.array(lrt_apply(rmsB1_slice, prev_window_val))

        else:
            start = time_sample[i]
            end = time_sample[i + 1]
            print(rms_df.rmsB1[start:end])

            rmsB1_slice = rms_df.rmsB1[start:end].copy().reset_index(drop=True)

            rms_df.rmsB1_LRT.iloc[start:end] = np.array(lrt_apply(rmsB1_slice, prev_window_val))

            prev_window_val = rms_df.rmsB1_LRT.iloc[end - 1]


def apply_gaussian1d_filter(rms_df, columnName, sigma=50):
    new_col_name = columnName + '_gauss_smooth'
    rms_df[new_col_name] = gaussian_filter1d(rms_df[columnName], sigma)


def plt_calc_rul_fig(rms_df, last_window, k_sum, n_post_tsp, postTSPCount, poly_fit, rmsSourceCol = 'rmsB1_gauss_smooth'):
    """
    Plots the RMS estimation and RUL values
    :param rms_df: main dataframe that holds the rms values
    :param last_window: the sample index of the last window where TSP is detected in
    :param k_sum: cumulative sum of predicted RMS values via quadratic poly fit, k param of eq9 of Ahmad's paper
    :param n_post_tsp: Number of sample point for the sliding window aftar TSP detected could be 40<= n_post_tsp<=60
    :param postTSPCount: Number of sample point where failure detected e.g. alpha >=0.0005
    :param poly_fit: is np.poly1d variable that includes quadratic poly fit to the post TSP windows
    :param rmsSourceCol: the source col name that the rms values will be plotted against e.g. rmsB1
    :return: Estimated RUL value
    """

    pred_rms_vals = poly_fit(rms_df.sample_indx[last_window: last_window + k_sum])
    maxPredRmsVal = max(pred_rms_vals)  # for comp. efficiency only do max once

    plt.figure(figsize=(10, 5))

    plt.plot(rms_df.sample_indx[0:last_window], rms_df[rmsSourceCol][0:last_window],
                 c='green', label='RMS values before TSP')

    plt.plot(rms_df.sample_indx[last_window: -1], rms_df[rmsSourceCol][last_window : -1],
             c='maroon', label='RMS values after TSP')

    plt.plot(rms_df.sample_indx[last_window: last_window + k_sum], pred_rms_vals,
             c='blue', linestyle='dashed', lw=2, label='Estimated RMS Trend')

    # plot a red line for bearing fault degradation
    plt.plot([postTSPCount + n_post_tsp, postTSPCount + n_post_tsp], [0, maxPredRmsVal], c='red', linestyle='-.')

    # plot a green line for TSP detection
    plt.plot([last_window, last_window], [0, maxPredRmsVal], c='green', linestyle='-.')

    # Place some text into the plot for TSP and Declaration of Failure
    plt.text(last_window + 2, maxPredRmsVal / 2, 'TSP detected', rotation=90)
    plt.text(last_window + k_sum + 2, maxPredRmsVal / 2, 'Declaration of Failure', rotation=90)
    plt.text(last_window + last_window / 10, maxPredRmsVal / 2,
             'The estimated RUL (min) {0}'.format((k_sum * 10), rotation=0))

    # Labels of the figure
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xticks(range(0, rms_df.shape[0], 50))
    plt.xlabel('Sample #')
    plt.ylabel('RMS Values')
    plt.show(block= False)
    plt.pause(3)
    plt.close()

    return k_sum * 10


def apply_linear_regr(x, y):

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return slope
