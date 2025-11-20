"""
Generates ACT cut objects (masks) for the simulated source TODs for mapmaking.
"""

import pandas as pd
import numpy as np
import os
os.environ['DOT_MOBY2']='/home/sn0543/.moby2'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import moby2
import cutslib as cl
import cutslib.glitch as gl
from moby2.tod import cuts
import sys
import argparse

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


sys.stdout.flush()
parser = argparse.ArgumentParser()

parser.add_argument("--datadir", type = str, help = "Name of directory with TODs and label lists", default = os.getcwd())

parser.add_argument("--outputdir", type = str, help = "Name of output directory", default = os.getcwd())

parser.add_argument("--toddir", type = str, help = "Name of TOD sims directory", default = os.getcwd())

parser.add_argument("--time_code", type = str, help = "Time code of depth 1 map", default = os.getcwd())

parser.add_argument("--half_life", type = str, help = "Half life of sim", required = True)

parser.add_argument("--amp", type = str, help = "Amplitude of sim", required = True)

parser.add_argument("--tod", type = str, help = "Name of TOD", required = True)

args = parser.parse_args()

args_dict = vars(args)

outdir = '{}/new_cuts_depth1/h{}/{}'.format(args_dict['outputdir'], args_dict['half_life'], args_dict['time_code'])

outdir_origsim = '{}/new_cuts_depth1_notmodified/h{}/{}'.format(args_dict['outputdir'], args_dict['half_life'],  args_dict['time_code'])

df_name = 'predicted_labels_df_sims_{}_amp{}_h{}_naffected_4'.format(args_dict['tod'], args_dict['amp'], args_dict['half_life'])

df = pd.read_csv('{}/{}.csv'.format(args_dict['datadir'], df_name))

tods = df.TOD.unique()

#get all glitches predicted to be a point source
df_ps = df.loc[df['Glitch Prediction'] == 0]

#get all high probability point sources 
df_ps_highprob = df_ps.loc[df_ps['Probability of being a Point Source'] > 0.7]

tod_high_probs = df_ps_highprob.TOD.unique()

if len(tod_high_probs) != 0:

    for t in tod_high_probs:

        df_temp = df_ps_highprob.loc[df_ps_highprob['TOD'] == t].reset_index()

        df_temp = df_temp.sort_values('Start Index')

        #get tslices for the glitches to be able to load them in and make new cuts objects
        tslices = np.zeros((len(df_temp), 2), dtype = int)
        
        if df_temp.shape[0] > 1:

            for i in range(len(df_temp)):
                tslices[i] = [int(df_temp['Start Index'][i]), int(df_temp['Stop Index'][i])]
                
        else:

            tslices[0] = [int(df_temp['Start Index'][0]), int(df_temp['Stop Index'][0])]
        
        #read in TOD and do simple pre-processing
        tod = cl.load_tod(t, depot = '/projects/ACT/yilung/depot/', release = '20230220')

        cl.quick_transform(tod, steps=['demean', 'detrend', 'ff_mce', 'cal', 'f_glitch'])

        #the follow portion reads in the sims to assign to the TOD object
        ctime_file = '{}/{}_amp{}_h{}_ctimes.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

        detid_file = '{}/{}_amp{}_h{}_detIDs.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

        tod_file = '{}/{}_amp{}_h{}.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

        ctimes = np.load(ctime_file)

        detids = np.load(detid_file)

        tod_sim = np.load(tod_file)

        det_uids_moby2 = [int(d.split("_")[-1]) for d in detids]
        
        #create dummy TOD with sims
        tod_dummy = DummyTOD.like_tod(tod, tod_sim, det_uids_moby2, ctimes)
        tod_cuts = cuts.get_glitch_cuts(tod = tod_dummy)

        cuts_t = cl.TODCuts.for_tod(tod, assign = False)

        #get original cuts object using our tslices that don't include high probability point sources
        exclude_cv = cl.CutsVector(tslices, tod_dummy.nsamps)
        buffer = 5

        #create cuts object per detector
        for det_uid in tod.det_uid:

            if det_uid in tod.cuts.get_cut():
                cuts_t.set_always_cut(det_uid)

            elif det_uid in tod_dummy.det_uid:
                
                ind_t = np.where(tod_dummy.det_uid == det_uid)[0][0]
                new_cv = cl.CutsVector.from_mask(exclude_cv.get_buffered(buffer).get_complement().get_mask() * tod_cuts.cuts[ind_t].get_mask())
                cuts_t.add_cuts(det_uid, new_cv)

            else:
                cuts_t.add_cuts(det_uid, tod.pcuts.cuts[det_uid])


        cuts_t.write_to_path('{}/{}_{}.cuts'.format(outdir, t[:-5], t[-4:]))

else:
    print('No TODs had a high probability point source!')

#make makes object for TOD with sims as the original saved ACT ones do not include the added sim information
for t in tods:

    df_temp = df.loc[df['TOD'] == t].reset_index()

    df_temp = df_temp.sort_values('Start Index')
    
    #read in TOD and do simple pre-processing
    tod = cl.load_tod(t, depot = '/projects/ACT/yilung/depot/', release = '20230220')

    cl.quick_transform(tod, steps=['demean', 'detrend', 'ff_mce', 'cal', 'f_glitch'])
    
    #the follow portion reads in the sims to assign to the TOD object
    ctime_file = '{}/{}_amp{}_h{}_ctimes.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

    detid_file = '{}/{}_amp{}_h{}_detIDs.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

    tod_file = '{}/{}_amp{}_h{}.npy'.format(args_dict['toddir'], t, args_dict['amp'], args_dict['half_life'])

    ctimes = np.load(ctime_file)

    detids = np.load(detid_file)

    tod_sim = np.load(tod_file)

    det_uids_moby2 = [int(d.split("_")[-1]) for d in detids]
    
    #create dummy TOD with sims
    tod_dummy = DummyTOD.like_tod(tod, tod_sim, det_uids_moby2, ctimes)
    tod_cuts = cuts.get_glitch_cuts(tod = tod_dummy)
    
    cuts_t = cl.TODCuts.for_tod(tod, assign = False)

    buffer = 5

    #make cuts object
    for det_uid in tod.det_uid:

        if det_uid in tod.cuts.get_cut():
            cuts_t.set_always_cut(det_uid)

        elif det_uid in tod_dummy.det_uid:
            ind_t = np.where(tod_dummy.det_uid == det_uid)[0][0]
            cuts_t.add_cuts(det_uid, tod_cuts.cuts[ind_t])

        else:
            cuts_t.add_cuts(det_uid, tod.pcuts.cuts[det_uid])


    cuts_t.write_to_path('{}/{}_{}.cuts'.format(outdir_origsim, t[:-5], t[-4:]))