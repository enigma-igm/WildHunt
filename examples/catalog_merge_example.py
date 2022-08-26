#!/usr/bin/env python



from wildhunt import catalog

from IPython import embed

print("Hallo")

if __name__ == '__main__':


    dtype = dtype={'comment': 'object',
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
    print(obs.df.columns)
    merged = candidates.catalog_cross_match(obs, 5, column_prefix='obs')

    embed()