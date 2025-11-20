'''
Functions for computing statistics for classifying and positions of glitches.
'''

import numpy as np
import scipy.stats as ss
import cutslib as cl
import cutslib.glitch as gl
from cutslib.visual import array_plots, get_position, tod3D
import moby2
from moby2.scripting import products
import scipy as sp


def num_of_det(x_pos):

    '''
    Compute the number of detectors affected by the glitch

    Input: x_pos: x positions across the focal plane computed from 
    cutslib.visual.get_position if using ACT data
    Output: number of detectors affected
    ''' 

    return len(x_pos)


def x_and_y_histogram_extent_ratio(x_pos, y_pos):

    '''
    Compute the ratio of the extents of the x position and y position 
    histograms.

    Input: x_pos: x positions and y_pos: y positions across the focal plane computed from 
    cutslib.visual.get_position if using ACT data
    Output: ratio of the extents of the x and y histograms
    ''' 

    hist_ratio = (np.max(y_pos) - np.min(y_pos))/(np.max(x_pos) - np.min(x_pos))

    return hist_ratio

def mean_time_lags(data):

    '''
    Compute the mean of the absolute value of the time lags between detectors.

    Input: data: detector TODs computed using data = snippets[s].data which 
    has been demeaned and detrended where s is the given snippet number
    Output: mean of the absolute value of the time lags
    ''' 

    lags = np.full((len(data), len(data)), np.nan)


    for i in range(len(data)):
            if len(data[i]) >= 2:
                for j in range(len(data)):
                    if j > i:

                        #compute the time delays between detector pair
                        time_delay_pos = np.fft.ifft(np.fft.fft(data[i])*np.conjugate(np.fft.fft(data[j])))

                        #find the maximum time delay which correspond to the time shift required to achieve the maximum correlation 
                        max_time_delay_pos = np.max(time_delay_pos)

                        time_delay_ind_t = np.where(time_delay_pos == max_time_delay_pos)[0][0]

                        #determine the difference between the required time shift and the length of the TOD
                        shift_t = time_delay_ind_t - len(data[i])

                        #Take the small between the required time shift and shift defined above. This allows for shifted backwards instead of looping around the TOD.
                        if np.abs(shift_t) < time_delay_ind_t:
                            lag_t = shift_t
                        else:
                            lag_t = time_delay_ind_t

                        lags[i, j] = lag_t

    time_lag = np.nanmean(np.abs(lags))

    return time_lag


def mean_correlation(data):

    '''
    Compute the mean of the absolute value of the Pearson correlation coefficient between detectors.

    Input: data: detector TODs computed using data = snippets[s].data which 
    has been demeaned and detrended where s is the given snippet number
    Output: mean of the absolute value of the correlations
    ''' 

    corr_coeff = np.full((len(data), len(data)), np.nan)

    for i in range(len(data)):
            if len(data[i]) >= 2:
                for j in range(len(data)):
                    if j > i:

                        #compute the Pearson correlation coefficient between detector pair
                        corr_t = ss.pearsonr(data[i], data[j])[0]

                        corr_coeff[i, j] = corr_t

    mean_corr = np.nanmean(np.abs(corr_coeff))

    return mean_corr


def max_and_near_y_pos_ratio(y_pos):

    '''
    Compute the ratio of the maximum of the y histogram bin and positions within 0.1 on either side of the focal plane
    compared to the total number of detectors.

    Input: y_pos: y positions across the focal plane computed from 
    cutslib.visual.get_position if using ACT data
    Output: sum of maximum and 0.1 to either side of the y histogram bins divided by the total number of detectors
    '''

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #find all bins within 0.1 of either side of the maximum bin
    ind_close_to_max = np.where(np.abs(np.histogram(y_pos)[1] - np.histogram(y_pos)[1][ind_y_max]) <= 0.1)[0]

    sum_close = np.sum(np.histogram(y_pos)[0][ind_close_to_max[:-1]])

    det_num = len(y_pos)

    return sum_close/det_num



def max_and_adjacent_y_pos_ratio(y_pos):

    '''
    Compute the ratio of the maximum and adjacent y histogram bins compared to the total number of detectors.

    Input: y_pos: y positions across the focal plane computed from 
    cutslib.visual.get_position if using ACT data
    Output: sum of maximum and adjacent y histogram bins divided by the total number of detectors
    '''

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #check if there are adjacent bins on either side of the maximun bin
    if ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1] + np.histogram(y_pos)[0][ind_y_max + 1]

    elif ind_y_max + 1 > len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1]

    elif ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 < 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max + 1]   
    
    
    det_num = len(y_pos)

    return sum_near/det_num


def compute_num_peaks(data):

    '''
    Computes the number of peaks in the combined TOD from the different detectors.

    Input: data: detector TODs computed using data = snippets[s].data which 
    has been demeaned and detrended where s is the given snippet number
    Output: the number of peaks
    ''' 

    #make a smoothing kernel
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size

    #smooth the data
    max_vals_t = np.convolve(np.max(data, axis = 0), kernel, mode='same')

    mean_vals_t = np.convolve(np.mean(data, axis = 0), kernel, mode='same')

    std_vals_t = np.std(data)

    vals_for_peaks = np.zeros(len(max_vals_t))

    #check if the max value is >= the mean *3std or else use the mean value
    for i in range(len(max_vals_t)):

        if max_vals_t[i] >= mean_vals_t[i] + 3*std_vals_t:
            vals_for_peaks[i] = max_vals_t[i]

        else:
            vals_for_peaks[i] = mean_vals_t[i]

    #find the peaks in the combined TOD
    prom = np.max([1e-12,  np.abs(np.mean(vals_for_peaks)) + 2.*np.mean(std_vals_t)])
    peaks_t = sp.signal.find_peaks(vals_for_peaks, prominence = prom)[0]

    num_peaks_t = len(peaks_t)

    return num_peaks_t


def ra_dec(tod_name, snip, start_ind, stop_ind, tod):

    '''
    Compute the median ctime, mean ra, and mean dec for the detectors seeing the glitch. 
    This is specific to the ACT snippet set up on tiger.

    Input: tod_name: name of the TOD, snip: snippet loaded with cutslib.glitch.affected_snippets_from_cv, 
    start_ind: glitch start tslice, stop_ind: glitch stop tslice, tod: tod loaded with cutslib.load_tod
    Output: median ctime, mean ra, and mean dec for the detectors seeing the glitch
    '''

    filenames = {'ar4': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar4_180303.txt', 
             'ar5': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar5_180303.txt',
             'ar6': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar6_180303.txt',
             'ar7': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar7_180303.txt'}
    
    array = tod_name[-8:-5]
    
    pointingpar = {
        'detector_offsets': {
           'format': 'fp_file',
           'filename': filenames[array]},
      }
    
    snip.fplane = products.get_focal_plane(pointingpar, 
    det_uid=snip.det_uid, tod_info=snip.info, tod=snip)
    ra, dec = moby2.pointing.get_coords(tod.ctime[range(start_ind, stop_ind)], 
    tod.az[range(start_ind, stop_ind)], tod.alt[range(start_ind, stop_ind)], 
    focal_plane=snip.fplane)

    det_ra = ra*180/np.pi
    det_dec = dec*180/np.pi

    #save the ra/dec at the peak for each detector
    d = snip.data
    ra_peaks = np.zeros(len(d))
    dec_peaks = np.zeros(len(d))

    # go through each detector and find ra/dec at the peak
    peaks = []
    for i in range (len(d)):
        val_peak = np.amax(d[i])
        peaks.append(val_peak)
        id_peak = np.where(d[i] == val_peak)[0][0]
        ra_peaks[i] = det_ra[i][id_peak]
        dec_peaks[i] = det_dec[i][id_peak]

    # average ra/dec at peak for each detector to get guess of PS position
    mean_ra = np.average(ra_peaks, weights=peaks)
    mean_dec = np.average(dec_peaks, weights=peaks)
    
    med_ctime = np.median(tod.ctime[range(start_ind, stop_ind)])

    return med_ctime, mean_ra, mean_dec

def ra_dec_w_TOD_offsets(tod_name, snip, start_ind, stop_ind, tod):

    '''
    Compute the mean ra, and mean dec for the detectors seeing the glitch with the TOD offsets. 
    This is specific to the ACT snippet set up on tiger.

    Input: tod_name: name of the TOD, snip: snippet loaded with cutslib.glitch.affected_snippets_from_cv, 
    start_ind: glitch start tslice, stop_ind: glitch stop tslice, tod: tod loaded with cutslib.load_tod
    Output: mean ra, and mean dec for the detectors seeing the glitch with the TOD offsets
    '''

    filenames = {'ar4': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar4_180303.txt', 
             'ar5': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar5_180303.txt',
             'ar6': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar6_180303.txt',
             'ar7': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar7_180303.txt'}
    
    shift_params = {
   'source':'file',
   'filename': '/projects/ACT/mhasse/depots/actpol_shared/TODOffsets/tod_offsets_dr6v4_20230223.txt',
   'columns': [0,3,4],
   'rescale_degrees': 1./60}
    
    array = tod_name[-8:-5]
    
    pointingpar = {
        'detector_offsets': {
           'format': 'fp_file',
           'filename': filenames[array]},
      }
    
    offset = np.deg2rad(products.get_pointing_offset(shift_params, tod=tod, source_offset=True))

    snip.fplane = products.get_focal_plane(pointingpar, 
    det_uid=snip.det_uid, tod_info=snip.info, tod=snip)
    ra, dec = moby2.pointing.get_coords(tod.ctime[range(start_ind, stop_ind)], 
        tod.az[range(start_ind, stop_ind)] - offset[0]/np.cos(tod.alt[range(start_ind, stop_ind)]), 
        tod.alt[range(start_ind, stop_ind)] - offset[1], 
        focal_plane=snip.fplane)

    ra_per_det = np.mean(ra, axis = 1)

    dec_per_det = np.mean(dec, axis = 1)

    mean_ra = np.mean(ra_per_det)
    mean_dec = np.mean(dec_per_det)
    

    return mean_ra, mean_dec


def sims_ra_dec(tod_name, snip, start_ind, stop_ind, tod, tod_dummy):

    '''
    Compute the median ctime, mean ra, and mean dec for the detectors seeing the glitch. 
    This is specific to the ACT snippet set up on tiger for simulated data.

    Input: tod_name: name of the TOD, snip: snippet loaded with cutslib.glitch.affected_snippets_from_cv, 
    start_ind: glitch start tslice, stop_ind: glitch stop tslice, tod: tod loaded with cutslib.load_tod,
    tod_dummy: dummy TOD with simulation metadata
    Output: median ctime, mean ra, and mean dec for the detectors seeing the glitch
    '''

    filenames = {'ar4': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar4_180303.txt', 
             'ar5': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar5_180303.txt',
             'ar6': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar6_180303.txt',
             'ar7': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar7_180303.txt'}
    
    array = tod_name[-8:-5]
    
    pointingpar = {
        'detector_offsets': {
           'format': 'fp_file',
           'filename': filenames[array]},
      }
    
    snip.fplane = products.get_focal_plane(pointingpar, 
    det_uid=tod_dummy.det_uid[snip.det_uid], tod_info=snip.info, tod=snip)
    ra, dec = moby2.pointing.get_coords(tod.ctime[range(start_ind, stop_ind)], 
    tod.az[range(start_ind, stop_ind)], tod.alt[range(start_ind, stop_ind)], 
    focal_plane=snip.fplane)

    ra_per_det = np.mean(ra, axis = 1)

    dec_per_det = np.mean(dec, axis = 1)

    mean_ra = np.mean(ra_per_det)
    mean_dec = np.mean(dec_per_det)
    
    med_ctime = np.median(tod.ctime[range(start_ind, stop_ind)])

    return med_ctime, mean_ra, mean_dec

def sims_ra_dec_w_TOD_offsets(tod_name, snip, start_ind, stop_ind, tod, tod_dummy):

    '''
    Compute the mean ra, and mean dec for the detectors seeing the glitch with the TOD offsets. 
    This is specific to the ACT snippet set up on tiger for simulated data.

    Input: tod_name: name of the TOD, snip: snippet loaded with cutslib.glitch.affected_snippets_from_cv, 
    start_ind: glitch start tslice, stop_ind: glitch stop tslice, tod: tod loaded with cutslib.load_tod,
    tod_dummy: dummy TOD with simulation metadata
    Output: mean ra, and mean dec for the detectors seeing the glitch with the TOD offsets
    '''

    filenames = {'ar4': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar4_180303.txt', 
             'ar5': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar5_180303.txt',
             'ar6': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar6_180303.txt',
             'ar7': '/projects/ACT/mhasse/depots/actpol_shared/RelativeOffsets/template_ar7_180303.txt'}
    
    shift_params = {
   'source':'file',
   'filename': '/projects/ACT/mhasse/depots/actpol_shared/TODOffsets/tod_offsets_dr6v4_20230223.txt',
   'columns': [0,3,4],
   'rescale_degrees': 1./60}
    
    array = tod_name[-8:-5]
    
    pointingpar = {
        'detector_offsets': {
           'format': 'fp_file',
           'filename': filenames[array]},
      }
    
    offset = np.deg2rad(products.get_pointing_offset(shift_params, tod=tod, source_offset=True))

    snip.fplane = products.get_focal_plane(pointingpar, 
    det_uid=tod_dummy.det_uid[snip.det_uid], tod_info=snip.info, tod=snip)
    ra, dec = moby2.pointing.get_coords(tod.ctime[range(start_ind, stop_ind)], 
        tod.az[range(start_ind, stop_ind)] - offset[0]/np.cos(tod.alt[range(start_ind, stop_ind)]), 
        tod.alt[range(start_ind, stop_ind)] - offset[1], 
        focal_plane=snip.fplane)

    ra_per_det = np.mean(ra, axis = 1)

    dec_per_det = np.mean(dec, axis = 1)

    mean_ra = np.mean(ra_per_det)
    mean_dec = np.mean(dec_per_det)
    

    return mean_ra, mean_dec
