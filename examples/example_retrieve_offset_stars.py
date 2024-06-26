#!/usr/bin/env python

"""
An example outlining on how to retrieve offset stars for an example catalog.

"""

import pandas as pd
from wildhunt import catalog


def retrieve_datalab_offsets(target_cat):
    """ Example function to retrieve offset stars from datalab services.

    :param target_cat: Target catalog
    :type target_cat: wildhunt.catalog.Catalog
    :return: None
    """

    # Create offset star catalog with datalab
    datalab_dict = {'table': 'ls_dr10.tractor',
                    'ra': 'ra',
                    'dec': 'dec',
                    'mag': 'mag_z',
                    'mag_name': 'lsdr10_z'}

    # The columns available for the query depend on the datalab table you are
    # querying.
    quality_query = 'mag_z > 15 AND mag_z < 19'

    # Retrieve the offset stars from the datalab table.
    target_cat.get_offset_stars_datalab(100, datalab_dict=datalab_dict,
                                        where=quality_query,
                                        minimum_distance=5)

    # Read the offset star catalog.
    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'example', 'ls_dr10'))
    # Print it df to screen for example purposes.
    print(df)


def retrieve_astroquery_offsets(target_cat):
    """ Example function to retrieve offset stars from astroquery services.

    :param target_cat: Target catalog
    :type target_cat: wildhunt.catalog.Catalog
    :return: None
    """

    # Define the quality query.
    # The query is written for a pandas query function using the column names
    # from the downloaded catalog columns.
    quality_query = 'distance > 3/60. and 10 < jAperMag3 < 18.5 ' \
                    'and jppErrBits==0'

    # Retrieve the offset stars from the UKIDSS DR11 catalog.
    target_cat.get_offset_stars_astroquery(100, catalog='ukidssdr11las',
                                           quality_query=quality_query,
                                           minimum_distance=5)

    # Read the offset star catalog.
    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'example', 'ukidssdr11las'))
    # Print it df to screen for example purposes.
    print(df)


def retrieve_ps1_offsets(target_cat):
    """ Example function to retrieve offset stars from Pan-STARRS1 services.

    :param target_cat: Target catalog
    :type target_cat: wildhunt.catalog.Catalog
    :return: None
    """

    # Create offset star catalog with Pan-STARRS1
    quality_query = 'iMeanPSFMag > 15 and iMeanPSFMag < 20.5'
    target_cat.get_offset_stars_ps1(100, quality_query=quality_query,
                                    catalog='mean')

    # Read the offset star catalog.
    df = pd.read_csv('{}_{}_OFFSETS.csv'.format(
        'example', 'ps1'))
    # Print it df to screen for example purposes.
    print(df)

if __name__ == "__main__":

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder.
    cat = catalog.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                          datapath='./data/pselqs_quasars_subset.csv')

    # Retrieve offset stars from datalab services
    retrieve_datalab_offsets(cat)

    # Retrieve offset stars from astroquery services
    retrieve_astroquery_offsets(cat)

    # Retrieve offset stars from Pan-STARRS1 services
    retrieve_ps1_offsets(cat)

