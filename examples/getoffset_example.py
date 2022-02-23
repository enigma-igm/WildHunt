#!/usr/bin/env python

import pandas as pd

from wildhunt import catalog as cat
from wildhunt import utils


df = pd.read_csv('UKIDSS_sources.csv')

target_name = df.loc[0, 'Name']
target_ra = df.loc[0, 'RA']
target_dec = df.loc[0, 'DEC']
radius = 100

catalog = 'ukidssdr11'

quality_query = 'distance > 3/60. and 10 < jAperMag3 < 18.5 and jppErrBits==0'

t = cat.get_astroquery_offset(target_name, target_ra, target_dec, radius,
                              catalog,
                              quality_query=quality_query, n=3, verbosity=2)

print(t)