import numpy as np
import os
os.environ['DOT_MOBY2']='/home/sn0543/.moby2'

from moby2.tod import cuts
from moby2.scripting import products
import cutslib as cl
import cutslib.glitch as gl
from cutslib.visual import array_plots, get_position, tod3D
import pandas as pd
import argparse
import multiprocessing as mp
import time

import filter_stats_functions as func
import sys

sys.stdout.flush()
parser = argparse.ArgumentParser()

parser.add_argument("--todfile", type = str, help = "Name of file containing TOD names", required = True)

parser.add_argument("--tod_sim_file", type=str, help = "Location of TOD sims", required = True)

parser.add_argument("--half_life", type = str, help = "Half life of sim", required = True)

parser.add_argument("--amp", type = str, help = "Amplitude of sim", required = True)

parser.add_argument("--datadir", type = str, help = "Name of directory with TODs and label lists", default = os.getcwd())

parser.add_argument("--outputdir", type = str, help = "Name of output directory", default = os.getcwd())

parser.add_argument("--output_df_name", type = str, help = "Name of output dataframe name", default = 'test_file_name')

parser.add_argument("--naffected", type = int, help = "Minimum number of detectors affected by glitch", default = 4)

parser.add_argument("--ACT", help = "Is this ACT data?", action = "store_true")

args = parser.parse_args()

args_dict = vars(args)

#If this is ACT data compute the ra/dec using ACT TOD data structure
if args_dict['ACT']:

    cols = ['Number of Detectors', 'Y and X Extent Ratio', 'Mean abs(Correlation)',
        'Mean abs(Time Lag)', 'Y Hist Max and Adjacent/Number of Detectors',
        'Within 0.1 of Y Hist Max/Number of Detectors', 'Number of Peaks',
        'TOD', 'Start Index', 'Stop Index', 'Start Ctime', 'Stop Ctime',
        'Median Ctime', 'ra', 'dec', 'ra with TOD Offsets', 'dec with TOD Offsets']
else:

    cols = ['Number of Detectors', 'Y and X Extent Ratio', 'Mean abs(Correlation)',
        'Mean abs(Time Lag)', 'Y Hist Max and Adjacent/Number of Detectors',
        'Within 0.1 of Y Hist Max/Number of Detectors', 'Number of Peaks',
        'TOD', 'Start Index', 'Stop Index', 'Start Ctime', 'Stop Ctime']

labels = ['Point Sources', 'Point Sources + Other', 'Cosmic Rays', 'Other']

#set up dummy class for TOD so you can set up the sims as a TOD object compatible with cutslib and moby2
class DummyInfo:
    def __init__(self, sample_index):
        self.sample_index = sample_index

class DummyTOD:
    def __init__(self, data, ctime, det_uid):
        self.data = data
        self.ctime = ctime
        self.det_uid = np.array(det_uid)
        self.abuses = []
        self.nsamps = len(ctime)

    @classmethod
    def like_tod(cls, tod, data, det_uid, ctime):
        obj = cls(data, ctime, det_uid)
        obj.info = tod.info
        return obj

def compute_all(tod_name):

    print('Starting TOD {}'.format(tod_name), flush = True)

    #the follow portion reads in the sims to assign to the TOD object
    ctime_file = '{}/{}_amp{}_h{}_ctimes.npy'.format(args_dict['tod_sim_file'], tod_name, args_dict['amp'], args_dict['half_life'])

    detid_file = '{}/{}_amp{}_h{}_detIDs.npy'.format(args_dict['tod_sim_file'], tod_name, args_dict['amp'], args_dict['half_life'])

    tod_file = '{}/{}_amp{}_h{}.npy'.format(args_dict['tod_sim_file'], tod_name, args_dict['amp'], args_dict['half_life'])

    ctimes = np.load(ctime_file)

    detids = np.load(detid_file)

    tod_sim = np.load(tod_file)

    det_uids_moby2 = [int(d.split("_")[-1]) for d in detids]

    #read in TOD and do simple pre-processing
    tod = cl.load_tod(tod_name, depot = '/projects/ACT/yilung/depot/', release = '20230220', abscal = '230313')

    cl.quick_transform(tod, steps=['demean', 'detrend', 'ff_mce', 'cal', 'abscal', 'f_glitch'])

    #create dummy TOD with sims
    tod_dummy = DummyTOD.like_tod(tod, tod_sim, det_uids_moby2, ctimes)
    tod_cuts = cuts.get_glitch_cuts(tod = tod_dummy, params={'buffer': 200})

    # get detectors that pass det cuts
    dets_sim = tod_cuts.get_uncut()
    # also get a mask for uncut detectors
    det_mask_sim = np.zeros(len(tod_dummy.det_uid), dtype=bool)
    det_mask_sim[dets_sim] = True

    # collapse all partial cuts to see how many detectors are cut at each time
    n_affected = np.sum([tod_cuts.cuts[det].get_mask() for det in dets_sim], axis=0)

    #if there are no glitches then just return nan
    if np.shape(n_affected) == np.shape(0.0) and int(n_affected) == 0:

        if args_dict['ACT']:

            info = np.empty(shape = (1, 17))
        else:

            info = np.empty(shape = (1, 12))
        
        info[0, 0] = np.nan
        info[0, 1] = np.nan
        info[0, 2] = np.nan
        info[0, 3] = np.nan
        info[0, 4] = np.nan
        info[0, 5] = np.nan
        info[0, 6] = np.nan
        info[0, 7] = np.nan
        info[0, 8] = np.nan
        info[0, 9] = np.nan
        info[0, 10] = np.nan
        info[0, 11] = np.nan

        if args_dict['ACT']:

            info[0, 12] = np.nan
            info[0, 13] = np.nan
            info[0, 14] = np.nan
            info[0, 15] = np.nan
            info[0, 16] = np.nan

    else:

        #create cuts vector which says what times more than 'naffacted' detectors have been flagged with a glitch
        cv = cl.CutsVector.from_mask((n_affected >= args_dict['naffected']))

        #make snippets of glitches which groups together samples from multiple detectors affected at the same time within 200 time samples
        snippets = gl.affected_snippets_from_cv(tod_dummy, tod_cuts, cv, det_mask_sim)

        if args_dict['ACT']:

            info = np.empty(shape = (len(snippets), 17))
        else:

            info = np.empty(shape = (len(snippets), 12))
        
        #compute summary statistics per snippet
        for s in range(len(snippets)):

            #remove the overall trends from the snippets
            s_t = snippets[s].demean()
            s_t = s_t.deslope()
            
            data = s_t.data

            slice_inds = snippets[s].tslice

            tstart, tstop = tod.ctime[[slice_inds.start, slice_inds.stop - 1]]

            pos_t, polfamily_t, freq_t = get_position(Detid = tod_dummy.det_uid[snippets[s].det_uid], instrument='actpol'
                                            , season = snippets[s].info.season, array = snippets[s].info.array)
            x_t, y_t = pos_t

            det_num = func.num_of_det(x_t)

            hist_ratio = func.x_and_y_histogram_extent_ratio(x_t, y_t)

            time_lag = func.mean_time_lags(data)

            corr = func.mean_correlation(data)

            near = func.max_and_near_y_pos_ratio(y_t)

            adjacent = func.max_and_adjacent_y_pos_ratio(y_t)

            num_peaks = func.compute_num_peaks(data)

            info[s, 0] = det_num
            info[s, 1] = hist_ratio
            info[s, 2] = corr
            info[s, 3] = time_lag
            info[s, 4] = adjacent
            info[s, 5] = near
            info[s, 6] = num_peaks
            info[s, 7] = 5
            info[s, 8] = slice_inds.start
            info[s, 9] = slice_inds.stop
            info[s, 10] = tstart
            info[s, 11] = tstop
            
            if args_dict['ACT']:

                #compute ra/dec
                med_ctime, mean_ra, mean_dec = func.sims_ra_dec(tod_name, snippets[s], slice_inds.start, 
                                                           slice_inds.stop, tod, tod_dummy)
                
                #compute ra/dec with detector offsets accounted for
                mean_ra_woffset, mean_dec_woffset = func.sims_ra_dec_w_TOD_offsets(tod_name, snippets[s], slice_inds.start, 
                                                           slice_inds.stop, tod, tod_dummy)

                info[s, 12] = med_ctime
                info[s, 13] = mean_ra
                info[s, 14] = mean_dec
                info[s, 15] = mean_ra_woffset
                info[s, 16] = mean_dec_woffset


    df = pd.DataFrame(np.asarray(info), columns = cols)

    df['Glitch'] = 'Unlabeled'

    df['TOD'] = tod_name

    print('Finished TOD {}'.format(tod_name), flush = True)

    return df

#read list of TODs to compute summary statistic for
tods = np.loadtxt('{}/{}'.format(args_dict['datadir'], args_dict['todfile']), dtype = str)
start = time.time()

if len(np.shape(tods)) == 0:

    df = compute_all('{}'.format(tods))

else:
    max_processes = mp.cpu_count()

    proccess_num = np.min([len(tods), max_processes])


    if __name__ == '__main__':

        with mp.Pool(processes = proccess_num) as pool:  
                                    
            dfs = pool.map(compute_all, tods)


    df = pd.concat(dfs)

df.dropna(inplace = True)

df.to_csv('{}/{}_naffected_{}.csv'.format(args_dict['outputdir'], args_dict['output_df_name'], args_dict['naffected']))

end = time.time()

total_time = end - start
print("\n"+ str(total_time))