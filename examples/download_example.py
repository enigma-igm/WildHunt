#!/usr/bin/env python

import pandas as pd

from wildhunt import image


df = pd.read_csv('UKIDSS_sources.csv')

survey_dict = [
               {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':50},
               {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['Y', 'J', 'H', 'K'],
                'fov':50},
               {'survey': 'DELSDR9', 'bands': ['g', 'r', 'z', '1', '2'],
                'fov':50}
                ]

image.get_images(df['RA'].values[:20], df['DEC'].values[:20], 'cutouts',
                 survey_dict,
                 n_jobs=400)
