#!/usr/bin/env python

import time
import pandas as pd

from wildhunt import catalog as whcat







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

    return df




def generate_finding_charts():
    """Create offset stars and finding charts for a catalog."""

    finding_chart_fov = 150

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder and using only the first 5 rows.
    df = pd.read_csv('./data/pselqs_quasars.csv')
    cat = whcat.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                        table_data=df[:100])

    offset_star_file = 'example_ls_dr10_OFFSETS.csv'

    cat.create_finding_charts(finding_chart_fov,
                              offset_star_file=offset_star_file,
                              survey='DELSDR9', band='z',
                              image_folder_path='./cutouts',
                              offset_focus=True, offset_focus_id=0,
                              offset_mag_colname='mag_z')



if __name__ == '__main__':


    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder.
    df = pd.read_csv('./data/pselqs_quasars_subset.csv')
    target_cat = whcat.Catalog('example', 'ps_ra', 'ps_dec',
                               'wise_designation', table_data=df[:100])

    # Retrieve offset stars
    offset_df = retrieve_datalab_offsets(target_cat)

    # Save the offset stars to a csv file
    offset_df.to_csv('example_ls_dr10_OFFSETS.csv', index=False)

    # Set the FOV for the finding chart
    finding_chart_fov = 150

    # Open the offset star catalog
    offset_star_file = 'example_ls_dr10_OFFSETS.csv'
    # Create the finding charts
    target_cat.create_finding_charts(finding_chart_fov,
                              offset_star_file=offset_star_file,
                              survey='DELSDR10', band='z',
                              image_folder_path='./cutouts',
                              offset_focus=True, offset_focus_id=0,
                              offset_mag_colname='mag_z')





