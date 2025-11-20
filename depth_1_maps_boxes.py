'''
Makes cutouts of depth-1 maps.
'''

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import Cutout2D
from pixell import enmap, enplot, reproject, utils, curvedsky 
import matplotlib.colors as colors

from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import argparse
import os

sys.stdout.flush()
parser = argparse.ArgumentParser()

parser.add_argument("--datadir", type = str, help = "Name of directory with cuts objects", default = os.getcwd())

parser.add_argument("--outputdir", type = str, help = "Name of output directory", default = os.getcwd())

parser.add_argument("--time_code", type = str, help = "Time code of depth 1 map", default = os.getcwd())

parser.add_argument("--half_life", type = str, help = "Half life of sim", required = True)

parser.add_argument("--amp", type = str, help = "Amplitude of sim", required = True)

parser.add_argument("--ra", type = float, help = "RA of source in degrees", required = True)

parser.add_argument("--dec", type = float, help = "Dec of source in degrees", required = True)

parser.add_argument("--depth_1", type = str, help = "Depth-1 file name", required = True)

args = parser.parse_args()

args_dict = vars(args)

depth1dir = '{}/sims_h{}_modifiedcuts/{}'.format(args_dict['datadir'], args_dict['half_life'], args_dict['time_code'])

depth1dir_origsim = '{}/sims_h{}/{}'.format(args_dict['datadir'], args_dict['half_life'],  args_dict['time_code'])

file_1 = '{}/{}_map.fits'.format(depth1dir, args_dict['depth_1'])

file_2 = '{}/{}_map.fits'.format(depth1dir_origsim, args_dict['depth_1'])

#read in maps with and without the simulated source.
imap_1 = enmap.read_map(file_1)

imap_2 = enmap.read_map(file_2)

def eshow(x, save = False, title = 'cutout', outdir = args_dict['outputdir'], **kwargs): 
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "ipython")

    if save:
        enplot.write('{}/{}'.format(outdir, title), plots)

def get_submap(imap, ra_deg, dec_deg, size_deg=0.5):
    ra = ra_deg*utils.degree
    dec = dec_deg*utils.degree
    radius = size_deg*utils.degree

    print(ra_deg, ra, dec_deg, dec, radius)
    omap = imap.submap([[dec - radius, ra - radius], [dec + radius, ra + radius]])
    return omap


keys = {"downgrade": 8, "ticks": 10, "colorbar": True}  

ra_deg, dec_deg = args_dict['ra'], args_dict['dec']

ra, dec = np.deg2rad([ra_deg, dec_deg])

# Give the box a width of n degrees
W = 2
width = np.deg2rad(W)

# Create the box and use it to select a submap
box = [[dec-width/2.,ra-width/2.],[dec+width/2.,ra+width/2.]]

#get cutouts of maps
smap_1 = imap_1[0].submap(box)

smap_2 = imap_2[0].submap(box)

plt.figure()
# Plot the map using the eshow function we defined earlier 
eshow(smap_2 - smap_1, save = True, outdir = args_dict["outputdir"], title = 'amp{}_h{}_{}_ra_{}_dec_{}_w_{}_sims-modifiedcuts'.format(args_dict['amp'], args_dict["half_life"], args_dict["depth_1"], ra_deg, dec_deg, W),  **{"colorbar":True})

plt.figure()
# Plot the map using the eshow function we defined earlier 
eshow(smap_1, save = True, outdir = args_dict["outputdir"], title = 'amp{}_h{}_{}_ra_{}_dec_{}_w_{}_modifiedcuts'.format(args_dict['amp'], args_dict["half_life"], args_dict["depth_1"], ra_deg, dec_deg, W),  **{"colorbar":True})

plt.figure()
eshow(smap_2, save = True, outdir = args_dict["outputdir"], title = 'amp{}_h{}_{}_ra_{}_dec_{}_w_{}_sims'.format(args_dict['amp'], args_dict["half_life"], args_dict["depth_1"], ra_deg, dec_deg, W),  **{"colorbar":True})
