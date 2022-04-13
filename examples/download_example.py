#!/usr/bin/env python

import time
import pandas as pd

from wildhunt import image

t0 = time.time()

df = pd.read_csv('UKIDSS_sources.csv')



survey_dict = [
                {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':50}
                #{'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'],
               #  'fov':120},
               #{'survey': 'DELSDR9', 'bands': ['z'],
               # 'fov':120}
                ]

image.get_images(df['RA'], df['DEC'], 'cutouts',
                 survey_dict, batch_size=10,
                 n_jobs=400)
print("{:.1f} s: ".format( time.time() - t0))