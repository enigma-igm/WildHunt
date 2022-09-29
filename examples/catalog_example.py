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
    # cat.df = cat.df.repartition(npartitions=10)

    # Test WSA
    # Reduce items in catalog
    cat.df = cat.df.sample(frac=0.1)
    print(cat.df.compute().shape)
    # Do the online cross-match
    # cat.online_cross_match(survey='UKIDSSDR11LAS')

    # Legacy Survey DecaLS
    cat.online_cross_match(survey='DELS')

    # UNWISE DR1
    cat.online_cross_match(survey='UNWISE')

    # CATWISE 2020
    cat.online_cross_match(survey='CATWISE')

