#!/usr/bin/env python

import pandas as pd

from wildhunt import image


df = pd.read_csv('UKIDSS_sources.csv')

survey_dict = [{'survey': 'ps1', 'bands': ['g', 'r'], 'fov':50},
              ]

image.get_images(df['RA'].values, df['DEC'].values, 'cutouts', survey_dict)

survey_dict = [{'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J', 'H'], 'fov':50},
              ]

image.get_images(df['RA'].values, df['DEC'].values, 'cutouts', survey_dict)