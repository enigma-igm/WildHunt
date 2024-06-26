#!/usr/bin/env python

"""
A simple example of how to perform an online cross-match between a catalog and
online catalogs.
"""

import pandas as pd
from wildhunt import catalog

if __name__ == "__main__":

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder.
    # Here we first load the catalog in a pandas dataframe and then inst
    # the catalog class with the dataframe as input.

    df = pd.read_csv('./data/pselqs_quasars.csv')

    # Selecting a random sample of 10 objects
    df = df.sample(10)

    cat = catalog.Catalog('example', ra_column_name='ps_ra',
                          dec_column_name='ps_dec',
                          id_column_name='wise_designation',
                          table_data=df)

    # Setting the verbosity to 2
    cat.verbose = 2

    # Perform pre-defined online cross-matches to
    # the UNWISE, DELS, UKIDSSDR11PLUSLAS.
    matched_cat_a = cat.online_cross_match(survey='UNWISE', match_distance=2)

    matched_cat_b = matched_cat_a.online_cross_match(survey='DELSDR9',
                                                     match_distance=2)

    matched_cat_c = matched_cat_b.online_cross_match(
        survey='UKIDSSDR11LAS', match_distance=2)

    matched_cat_c.name = 'example_cross_match_pselqs'

    matched_cat_c.save_catalog(output_dir='./catalogs')

    from Ipython import embed

    embed()
