#!/usr/bin/env python

""" Example scripts to show how to create a finding chart with WildHunt.

To successfully run the get_survey_images() function, you will need to have
generated the offset star files. Have a look at the example script for
retrieving offset stars (example_retrieve_offset_stars.py) for more details.

"""

import pandas as pd
from wildhunt import catalog as whcat
from wildhunt import image as whim
import matplotlib.pyplot as plt


def show_single_finding_chart():
    """Show a single finding chart for a single target."""

    finding_chart_fov = 150

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder and using only the first 5 rows.
    df = pd.read_csv('./data/pselqs_quasars.csv')
    cat = whcat.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                        table_data=df[:5])

    # Download the images
    fov = finding_chart_fov * 2  # field of view in arcseconds
    survey_dict = [
        {'survey': 'PS1', 'bands': ['i'], 'fov': fov}]

    cat.get_survey_images('./cutouts', survey_dict, n_jobs=5)

    # Get the offset stars
    # Create offset star catalog with Pan-STARRS1
    quality_query = 'iPSFMag > 15 and iPSFMag < 19 and distance > ' \
                    '3/3600'
    cat.get_offset_stars_ps1(finding_chart_fov / 2,
                             quality_query=quality_query,
                             n=5)

    offsets = pd.read_csv('example_ps1_OFFSETS.csv')

    # Choosing a target index from the catalog for example purposes
    idx = 2

    # Selecting the offsets relevant to that target
    target_ra = df.loc[idx, 'ps_ra']
    target_dec = df.loc[idx, 'ps_dec']
    target_name = df.loc[idx, 'wise_designation']

    offset_target = offsets.query('{}=="{}"'.format('target_name',
                                                    target_name))

    # Open the survey image
    img = whim.SurveyImage(df.loc[idx, 'ps_ra'], df.loc[idx, 'ps_dec'],
                           survey='PS1', band='i', image_dir='./cutouts',
                           min_fov=fov)

    # Plot the finding chart
    img.finding_chart(finding_chart_fov, target_aperture=5,
                      offset_df=offset_target,
                      # If you want to focus on the offset star uncomment
                      offset_focus=True, offset_id=offset_target.index[0],
                      offset_mag_column_name='ps1_dr2_stack_psfmag_y',
                      color_scale='zscale')
    # Show the plot
    plt.show()


def generate_finding_charts():
    """Create offset stars and finding charts for a catalog."""

    finding_chart_fov = 150

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder and using only the first 5 rows.
    df = pd.read_csv('./data/pselqs_quasars.csv')
    cat = whcat.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                        table_data=df[:100])

    offset_star_file = 'example_ps1_OFFSETS.csv'

    cat.create_finding_charts(finding_chart_fov,
                              offset_star_file=offset_star_file,
                              survey='PS1', band='i',
                              image_folder_path='./cutouts',
                              offset_focus=True, offset_focus_id=0,
                              offset_mag_colname='ps1_dr2_stack_psfmag_y')

    offset_star_file = 'example_ls_dr9_OFFSETS.csv'

    # cat.create_finding_charts(finding_chart_fov,
    #                           offset_star_file=offset_star_file,
    #                           survey='DELSDR9', band='z',
    #                           image_folder_path='./cutouts',
    #                           offset_focus=True, offset_focus_id=0,
    #                           offset_mag_colname='mag_z')

    # offset_star_file = 'example_ukidssdr11las_OFFSETS.csv'
    #
    # cat.create_finding_charts(finding_chart_fov,
    #                           offset_star_file=offset_star_file,
    #                           survey='UKIDSSDR11PLUSLAS', band='J',
    #                           image_folder_path='./cutouts',
    #                           offset_focus=False, offset_focus_id=0,
    #                           offset_mag_colname='jAperMag3')

if __name__ == '__main__':

    show_single_finding_chart()

    # generate_finding_charts()