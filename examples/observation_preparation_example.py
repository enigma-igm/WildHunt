#!/usr/bin/env python

import pandas as pd

from wildhunt import catalog_queries as whcat
from wildhunt import findingchart as whfc
from wildhunt import image as whim

# Load candidate list

candidates = pd.read_csv('LRIS_VI_candidates.csv')


# Query for offset stars using

# Define datalab dictionary
datalab_dict = {'survey': 'ls_dr9',
                'table': 'tractor',
                'ra': 'ra',
                'dec': 'dec',
                'mag': 'mag_r',
                'mag_name': 'lsdr9_r'}

# Radius should be adjusted to FOV of acquisition camera
radius = 50

# The magnitude filter band and range should be adjusted regarding the filter
# used for acquisition. Also, it is wise to only select point sources as
# offset stars.
where = "mag_r > 1 AND mag_r < 20 and type='PSF'"

# Get the offset star table
offsets = whcat.get_offset_stars_datalab(candidates, 'Name', 'RA', 'DEC',
                                         radius,
                                         datalab_dict, where=where,
                                         verbosity=2)

offsets.to_csv('offset_stars.csv')

# Download the cutouts
survey_dict = [
               {'survey': 'DELSDR9', 'bands': ['r'],
               'fov':110}
                ]

whim.get_images(candidates['RA'].values,
                 candidates['DEC'].values,
                 'cutouts',
                 survey_dict,
                 n_jobs=8)


# Create the finding charts
ra_column_name = 'RA'
dec_column_name = 'DEC'
target_column_name = 'Name'
survey = 'DELSDR9'
band = 'r'
aperture = 2
fov = 110
image_folder_path = './cutouts'

offsets = pd.read_csv('offset_stars.csv')

whfc.make_finding_charts(candidates, ra_column_name, dec_column_name,
                         target_column_name, survey, band,
                         aperture, fov, image_folder_path,
                         offset_table=offsets,
                         offset_id=0,
                         offset_focus=False,
                         offset_ra_column_name='offset_ra',
                         offset_dec_column_name='offset_dec',
                         pos_angle_column_name='pos_angle',
                         offset_mag_column_name='mag_r',
                         offset_id_column_name='offset_shortname',
                         format='png',
                         # slit_width=5,
                         # slit_length=100,
                         auto_download=False)


