#!/usr/bin/env python

import pandas as pd
from wildhunt import catalog as whcat
from wildhunt import image as whim
import matplotlib.pyplot as plt

from IPython import embed

if __name__ == '__main__':

    finding_chart_fov = 150

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder and using only the first 5 rows.
    df = pd.read_csv('./data/pselqs_quasars.csv')
    cat = whcat.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                        table_data=df[:5])

    # Download the images
    fov = finding_chart_fov * 2  # field of view in arcseconds
    survey_dict = [
        {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov': fov}]

    cat.get_survey_images('./cutouts', survey_dict, n_jobs=5)

    # Get the offset stars
    # Create offset star catalog with Pan-STARRS1
    quality_query = 'iPSFMag > 15 and iPSFMag < 19 and distance > ' \
                    '3/3600'
    cat.get_offset_stars_ps1(finding_chart_fov/2,
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
                      offset_mag_column_name='ps1_dr2_stack_psfmag_y')
    # Show the plot
    plt.show()
