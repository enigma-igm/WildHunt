#!/usr/bin/env python

from wildhunt.instruments import instrument_observations_setup

instrument_observations_setup('vlt', 'fors2', table = 'observation_starlist_catalog.csv',
                              ra_column_name = 'RA_deg', dec_column_name = 'DEC_deg', genertate_fcs = False,
                              target_column_name = 'Name', mag_column_name = 'mag', offset_ra_column_name = 'offset_ra',
                              offset_dec_column_name = 'offset_dec', offset_mag_column_name = 'offset_mag',
                              output_starlist = 'test_starlist', pos_angle_column_name = None, survey = 'DELSDR9',
                              band = 'r', aperture = 2, fov = 120, image_folder_path ='../cutouts', n_jobs = 5)