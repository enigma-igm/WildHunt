#!/usr/bin/env python

"""
A simple example for calculating aperture photometry for a catalog of sources.

"""

import pandas as pd
from wildhunt import catalog

from IPython import embed

if __name__ == "__main__":
    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder.
    # Here we first load the catalog in a pandas dataframe and then inst
    # the catalog class with the dataframe as input.

    df = pd.read_csv('./data/pselqs_quasars.csv')

    cat = catalog.Catalog('example', ra_column_name='ps_ra',
                          dec_column_name='ps_dec',
                          id_column_name='wise_designation',
                          table_data=df)

    # Definding a list of dictionaries containing the survey information.
    fov = 120  # field of view in arcseconds

    survey_dicts = [
        {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':fov},
        {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'], 'fov':fov},
        # Our current implementation of DELS does not correctly download images
        # at the moment.
        # {'survey': 'DELSDR9', 'bands': ['z'], 'fov': fov}
    ]

    ap_phot_cat = cat.get_forced_photometry(
        survey_dicts, image_folder_path='./cutouts',
        output_path='./catalogs/aperture_photometry_example', n_jobs=5)


