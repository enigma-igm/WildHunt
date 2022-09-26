#!/usr/bin/env python

import pandas as pd

from wildhunt import catalog as whcat
from wildhunt import findingchart

if __name__ == '__main__':
    df = pd.read_csv('LRIS_VI_candidates.csv')

    # Load example catalog
    cat = whcat.Catalog('qso_candidates',
                        ra_column_name='RA',
                        dec_column_name='DEC',
                        id_column_name='Name',
                        table_data=df)

    # Create offset star catalog with datalab
    datalab_dict = {'table': 'ls_dr9.tractor',
                    'ra': 'ra',
                    'dec': 'dec',
                    'mag': 'mag_r',
                    'mag_name': 'lsdr9_r'}

    quality_query = "mag_r > 15 AND mag_r < 20.5 AND type = 'PSF'"

    cat.get_offset_stars_datalab(100, datalab_dict=datalab_dict,
                                 where=quality_query)

    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'qso_candidates', datalab_dict['table'].split('.')[0]))

    '''
    # Create offset star catalog with PS1

    quality_query = "yMeanPSFMag > 15 and yMeanPSFMag < 19"

    cat.get_offset_stars_ps1(100, quality_query=quality_query)

    df = pd.read_csv('qso_candidates_ps1_OFFSETS.csv')
    '''
    # Download the cutouts
    survey_dict = [
        {'survey': 'DELSDR9', 'bands': ['r'],
         'fov': 150}
    ]

    cat.get_survey_images('cutouts', survey_dict, n_jobs=10)

    # Create the finding charts
    ra_column_name = 'target_ra'
    dec_column_name = 'target_dec'
    target_column_name = 'target_name'
    survey = 'DELSDR9'
    band = 'r'
    aperture = 2
    fov = 150
    image_folder_path = './cutouts'

    offsets = pd.read_csv('qso_candidates_ls_dr9_OFFSETS.csv')

    findingchart.make_finding_charts(df, ra_column_name, dec_column_name,
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

