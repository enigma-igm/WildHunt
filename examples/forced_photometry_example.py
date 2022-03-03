#!/usr/bin/env python

import time
import pandas as pd

from wildhunt import image, photometry

from IPython import embed

t0 = time.time()

df = pd.read_csv('UKIDSS_sources.csv')



survey_dict = [
                #{'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':50},
               {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['Y','J','H','K'], 'fov':120},
               #{'survey': 'DELSDR9', 'bands': ['g','r','z','1','2'], 'fov':120}
                ]

image.get_images(df['RA'].values[:10], df['DEC'].values[:10], 'cutouts',survey_dict,n_jobs=400)

forced_photometry=photometry.Forced_photometry(survey_dict[0]['bands'],survey_dict[0]['fov'],survey_dict[0]['survey'])
#forced_photometry.forced_main(df['RA'].values[:10], df['DEC'].values[:10],'test',image_folder_path='cutouts')

print("{:.1f} s: ".format( time.time() - t0))