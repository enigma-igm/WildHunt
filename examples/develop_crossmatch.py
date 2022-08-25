
import os, getpass

import pandas as pd

from dl import authClient as ac, queryClient as qc, storeClient as sc
from dl.helpers.utils import convert


def datalab_xmatch_example():
    ls_dr9_default_columns = """
          match.objid,
          match.ls_id,
          match.ra,
          match.dec,
          match.flux_g, match.flux_ivar_g,
          match.flux_r, match.flux_ivar_r,
          match.flux_z, match.flux_ivar_z,
          match.flux_w1, match.flux_ivar_w1,
          match.flux_w2, match.flux_ivar_w2,
          match.flux_w4, match.flux_ivar_w3,
          match.flux_w3, match.flux_ivar_w4,
          match.mag_g,
          match.mag_r,
          match.mag_z,
          match.mag_w1,
          match.mag_w2,
          match.mag_w3,
          match.mag_w4,
          match.maskbits,
          match.snr_g,
          match.snr_r,
          match.snr_z,
          match.snr_w1,
          match.snr_w2,
          match.snr_w3,
          match.snr_w4,
          match.wisemask_w1,
          match.wisemask_w2,
          match.type, 
          match.ebv, 
      """

    # FIRST STEP LOG INTO NOIRLAB database

    token = ac.login(input('Enter user name (+ENTER): '),
                     getpass.getpass('Enter password (+ENTER): '))

    print('User: ' + ac.whoAmI())

    print("Listing mydb tables:\n" + qc.mydb_list())

    # IMPORT local data table into datalab MyDB

    df = pd.read_csv('UKIDSS_sources.csv')

    print('Pandas result: ' + qc.mydb_import('upload', df))

    print("Listing mydb tables:\n" + qc.mydb_list())

    # Now make a cross-match query

    columns = ls_dr9_default_columns

    # columns = None

    designation_colname = 'name'
    ra_colname = 'ra'
    dec_colname = 'dec'

    upload_table = 'mydb://upload'
    survey = 'ls_dr9'
    table = 'tractor'

    # Build SQL query

    if columns is not None:
        sql_query = '''SELECT s.{} as source_designation,
                         s.{} as source_ra,
                         s.{} as source_dec, '''.format(designation_colname,
                                                        ra_colname,
                                                        dec_colname)
        sql_query += '{}'.format(columns)
    else:
        sql_query = 'SELECT * '

    sql_query += \
        '(q3c_dist(s.ra, s.dec, match.ra, match.dec)*3600) as dist_arcsec '

    sql_query += 'FROM {} AS s '.format(upload_table)

    sql_query += 'LEFT JOIN LATERAL ('

    sql_query += 'SELECT g.* FROM {}.{} AS g '.format(survey, table)

    sql_query += 'WHERE q3c_join(s.ra, s.dec, g.ra, g.dec, 0.01) '

    sql_query += 'ORDER BY q3c_dist(s.ra,s.dec,g.ra,g.dec) ASC LIMIT 1) '

    sql_query += 'AS match on true'

    # Query DataLab and write result to temporary file
    # if verbosity > 0:
    print("SQL QUERY: {}".format(sql_query))

    response = qc.query(sql=sql_query, fmt='csv')

    result_df = convert(response)

    print(result_df)


    # TODO DROPPING TABLES

    # # Drop any tables or temp files we created.
    # if do_cleanup:
    #     for table in ['test1', 'test2', 'objects1', 'objects2', 'objects3',
    #                   'objects4', 'objects5', 'objects6a', 'objects6b',
    #                   'objects6c']:
    #         print(
    #             "Dropping table '" + table + "' ... \t" + qc.mydb_drop(table))
    #     print("Listing mydb tables:\n" + qc.mydb_list())
    #
    #     for f in ['objects.csv', 'schema.txt']:
    #         if os.path.exists(f):
    #             os.remove(f)
    #     if sc.access('vos://objects.csv'):
    #         print('removing objects.csv')
    #         sc.rm('vos://objects.csv')
    #     if sc.access('vos://objects.vot'):
    #         print('removing objects.vot')
    #         sc.rm('vos://objects.vot')


if __name__ == '__main__':

    # DATALAB also has allwise, unwise_dr1, ukidss_dr11+ and vhs_dr5

    datalab_xmatch_example()