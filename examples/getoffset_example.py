#!/usr/bin/env python

import pandas as pd
from wildhunt import catalog as whcat


if __name__ == '__main__':

    df = pd.read_csv('UKIDSS_sources.csv')[:10]

    # Load example catalog
    cat = whcat.Catalog('ukidss_example',
                        ra_column_name='RA',
                        dec_column_name='DEC',
                        id_column_name='Name',
                        table_data=df)


    # Create offset star catalog with datalab
    datalab_dict = {'table': 'ls_dr9.tractor',
                        'ra': 'ra',
                        'dec': 'dec',
                        'mag': 'mag_z',
                        'mag_name': 'lsdr9_z'}

    quality_query = 'mag_z > 15 AND mag_z < 19'

    cat.get_offset_stars_datalab(100, datalab_dict=datalab_dict,
                                 where=quality_query)

    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'ukidss_example', datalab_dict['table'].split('.')[0]))

    print(df)

    # Create offset star catalog with astroquery
    quality_query = 'distance > 3/60. and 10 < jAperMag3 < 18.5 and jppErrBits==0'

    cat.get_offset_stars_astroquery(100, catalog='ukidssdr11')

    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'ukidss_example', 'ukidssdr11'))

    print(df)
