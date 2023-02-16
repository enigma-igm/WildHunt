#!/usr/bin/env python

import pandas as pd

from wildhunt import catalog_queries as whcq

df = pd.read_csv('UKIDSS_sources.csv')

ra = df.loc[0, 'RA']
dec = df.loc[0, 'DEC']


# VIKING example
# service = 'VSA'
# catalog = 'VIKING'
# data_release = 'VIKINGDR5'
#
# radius = 5
#
# t = cat.query_region_astroquery(ra, dec, radius, service, catalog,
#                             data_release=data_release)
#
# # UKIDSS example
#
# service = 'UKIDSS'
# catalog = 'LAS'
# data_release = 'UKIDSSDR11PLUS'
#
# t = cat.query_region_astroquery(ra, dec, radius, service, catalog,
#                             data_release=data_release)
#
# print(t)
#

# NOIRLAB Datalab Legacy Survey DR9 example

where = 'mag_z > 15 AND mag_z < 21'
ra = 168.1936986
dec = -3.6324306
radius = 30

datalab_dict = {'survey': 'ls_dr9',
                'table': 'tractor'}

t = whcq.query_region_datalab(ra, dec, radius, datalab_dict,
                         columns=None, where=None,
                             minimum_distance=3./60, verbosity=0)

