from utils import *

if __name__ == "__main__":
    rms_df = prepare_data()

    # plot the raw RMS bearing data and save the fig into the results folder
    plot_rms_vals(rms_df, columnName= 'rmsB1')
    plot_rms_vals(rms_df, columnName= 'rmsB2')
    plot_rms_vals(rms_df, columnName= 'rmsB3')
    plot_rms_vals(rms_df, columnName= 'rmsB4')

    # plot kurtosis
    plot_rms_vals(rms_df, columnName= 'kurtosis1')
    # plot the skew
    plot_rms_vals(rms_df, columnName= 'skew1')

    #plot the stdn1
    plot_rms_vals(rms_df, columnName='stdn1')

    plot_rms_vals(rms_df, columnName='mean1')



    apply_gaussian1d_filter(rms_df, 'rmsB1', sigma=50)

    rms_df['sample_indx'] = rms_df.index

    # number of samples to be taken for the sliding window, might change post TSP according to the paper
    # thus two variables
    n = 50
    n_post_tsp = 50

    # the slopes aka gradients of the fits according to the reference paper
    tsp_threshold = 0.0001
    failure_threshold = 0.0005

    last_window = 0

    for i in range(0, rms_df.shape[0], n):

        slope = apply_linear_regr(rms_df.sample_indx[i:i + n],
                                  rms_df['rmsB1_gauss_smooth'][i:i + n])
        print("TSP Slope for window index {0}: {1}".format(i, slope))
        if slope >= tsp_threshold:
            last_window = i + n
            print('TSP detected...')
            print('Last window index {0}'.format(last_window))
            break

    '''
    fit a quadratic (2nd order) formula to window after TSP
    has been detected 
    where last_window is the index of sample number that TSP
    has been determined
    '''
    poly_fit = np.poly1d(np.polyfit(rms_df.sample_indx[last_window:last_window + n_post_tsp],
                                    rms_df['rmsB1_gauss_smooth'][last_window:last_window + n_post_tsp], 2))

    # a counter for the RUL calculation eq. 9
    # know as the number of predicted RMS values before reaching the threshold
    k_sum = 0

    for postTSPCount in range(last_window, rms_df.shape[0], n_post_tsp):
        k_sum += n_post_tsp

        # evaluate the quadratic poly fit and fit a linear regression to get the slope
        slope = apply_linear_regr(rms_df.sample_indx[postTSPCount: postTSPCount + n_post_tsp],
                                  poly_fit(rms_df.sample_indx[postTSPCount: postTSPCount + n_post_tsp]))

        if slope >= failure_threshold:
            print("slope for the quad fit starting at index {0} is {1}".format(postTSPCount, slope))
            print('Threshold detected ...')
            print('k value {}'.format(k_sum))
            break

    # plot_rms_vals(rms_df, 'rmsB1_gauss_smooth' )
    # plot_rms_vals(rms_df, 'rmsB1')

    estimatedRUL = plt_calc_rul_fig(rms_df, last_window, k_sum, n_post_tsp, postTSPCount, poly_fit)
    estimatedRUL = plt_calc_rul_fig(rms_df, last_window, k_sum, n_post_tsp, postTSPCount, poly_fit,
                                    rmsSourceCol='rmsB1')
    print("The RUL estimation for this dataset: {0}".format(estimatedRUL))
