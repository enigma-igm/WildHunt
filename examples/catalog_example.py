#!/usr/bin/env python

from wildhunt import catalog

if __name__ == '__main__':

    filename = 'UKIDSS_sources.csv'
    ra_colname = 'RA'
    dec_colname = 'DEC'


    # Instantiate a catalog from a csv file
    cat = catalog.Catalog('example', ra_column_name=ra_colname,
                              dec_column_name=dec_colname,
                              id_column_name='Name',
                              datapath=filename)

    # Cross-match catalog to DELS survey.

    cat.online_cross_match(survey='DELS')


