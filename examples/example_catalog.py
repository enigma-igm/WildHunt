#!/usr/bin/env python

import time

from wildhunt import catalog


def example_online_cross_match():

    filename = 'data/pselqs_quasars.csv'
    ra_colname = 'ps_ra'
    dec_colname = 'ps_dec'
    id_colname = 'wise_designation'

    # filename = 'data/UKIDSS_sources_subset.csv'
    # ra_colname = 'RA'
    # dec_colname = 'DEC'
    # id_colname = 'Name'


    # Instantiate a catalog from a csv file
    cat = catalog.Catalog('example', ra_column_name=ra_colname,
                              dec_column_name=dec_colname,
                              id_column_name=id_colname,
                              datapath=filename)

    cat.verbose = 0

    # Online cross-match presets:

    # UKIDSS DR11 LAS
    cat.online_cross_match(survey='UKIDSSDR11LAS',
                           output_dir='./catalogs')

    # Legacy Survey DecaLS
    cat.online_cross_match(survey='DELSDR10',
                           output_dir='./catalogs')

    # UNWISE DR1
    cat.online_cross_match(survey='UNWISE',
                           output_dir='./catalogs')

    # CATWISE 2020
    cat.online_cross_match(survey='CATWISE',
                           output_dir='./catalogs')

    # Online cross-match to datalab table:
    cat.online_cross_match(survey=None,
                           output_dir='./catalogs',
                           astro_datalab_table='allwise.source')


    # Online cross-match to astroquery resource:
    cat.online_cross_match(survey=None,
                           output_dir='./catalogs',
                           astroquery_service='VSA',
                           astroquery_catalog='VHS',
                           astroquery_dr='VHSDR6')

def example_offline_cross_match():

    dtype = {'comment': 'object',
       'obs_date': 'object'}

    candidates = catalog.Catalog('candidates',
                                     datapath='data/reclassified_all_observations_upto1901_clean.csv',
                                     ra_column_name='obs_sdss_ra',
                                     dec_column_name='obs_sdss_dec',
                                     id_column_name='wise_designation',
                                     dtype=dtype)

    candidates.df = candidates.df.repartition(10)

    dtype = {'FIRST_SD_Cl': 'object',
             'FIRST_field_name': 'object',
             'mq_xname': 'object',
             'simbad_id': 'object',
             'simbad_ref': 'object',
             'simbad_type': 'object'}

    obs = catalog.Catalog('observed',
                              datapath='data/new_elqs_quasars.csv',
                              ra_column_name='wise_ra_x',
                              dec_column_name='wise_dec_x',
                              id_column_name='wise_designation',
                              dtype=dtype)

    candidates.catalog_cross_match(obs, 5, column_prefix='obs')



def example_download_images():

    t0 = time.time()

    cat = catalog.Catalog('example', 'RA', 'DEC','Name',
                          datapath='./data/UKIDSS_sources.csv')

    survey_dict = [
        {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':50},
         {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'],
          'fov':120},
        {'survey': 'DELSDR10', 'bands': ['z'],
         'fov': 120},
        {'survey': 'allWISE', 'bands': ['3', '4'], 'fov': 120}
    ]

    cat.get_survey_images('./cutouts',  survey_dict, n_jobs=3)
    print("{:.1f} s: ".format(time.time() - t0))



if __name__ == '__main__':

    example_online_cross_match()

    # example_download_images()

    # # Cross-match catalog to DELS survey.
    # # cat.df = cat.df.repartition(npartitions=10)
    #
    # # Test WSA
    # # Reduce items in catalog
    # cat.df = cat.df.sample(frac=0.1)
    # print(cat.df.compute().shape)
    # # Do the online cross-match
    # # cat.online_cross_match(survey='UKIDSSDR11LAS')
    #