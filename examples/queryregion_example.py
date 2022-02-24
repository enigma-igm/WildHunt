#!/usr/bin/env python

import pandas as pd

from wildhunt import catalog as cat

df = pd.read_csv('UKIDSS_sources.csv')

ra = df.loc[0, 'RA']
dec = df.loc[0, 'DEC']

service = 'VSA'
catalog = 'VIKING'
data_release = 'VIKINGDR5'

radius = 5

t = cat.query_region_astroquery(ra, dec, radius, service, catalog,
                            data_release=data_release)

# for col in t.columns:
#     print(col)
print(t)

service = 'UKIDSS'
catalog = 'LAS'
data_release = 'UKIDSSDR11PLUS'

t = cat.query_region_astroquery(ra, dec, radius, service, catalog,
                            data_release=data_release)

# for col in t.columns:
#     print(col)

print(t)