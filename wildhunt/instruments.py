#!/usr/bin/env python
"""

Main module for generating the files required to perform observations at the telescope

"""
import os

import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord

from wildhunt.utils import coord_to_name, degree_to_hms
from wildhunt import image as whim
from wildhunt import findingchart as whfc

from IPython import embed

def instrument_observations_setup(instrument, table, ra_column_name, dec_column_name, target_column_name,
                                  mag_column_name, offset_ra_column_name, offset_dec_column_name,
                                  offset_mag_column_name, output_starlist, pos_angle_column_name, survey, band,
                                  aperture = 2, fov = 110, image_folder_path = './cutouts', n_jobs = 4):

    if instrument == 'keck_lris':
        keck_lris(table, ra_column_name, dec_column_name, target_column_name, mag_column_name,
                  offset_ra_column_name, offset_dec_column_name, offset_mag_column_name, output_starlist,
                  pos_angle_column_name, survey, band, aperture, fov, image_folder_path, n_jobs)

    if instrument == None:
        print('ERROR: instrument not yet implemented')

# Below are defined the functions to prepare the observations for the different instruments

def keck_lris(table, ra_column_name, dec_column_name, target_column_name, mag_column_name,
              offset_ra_column_name, offset_dec_column_name, offset_mag_column_name, output_starlist,
              pos_angle_column_name = None, survey = 'DELSDR9', band = 'r', aperture = 2, fov = 110,
              image_folder_path = './cutouts', n_jobs = 4):

    # Open the candidates catalog
    df = pd.read_csv(table)

    output_folder = os.getcwd()

    # Defining the name of the targets and convert the RA, DEC from degrees to HMS, DMS
    ra_targ = df[ra_column_name].values
    dec_targ = df[dec_column_name].values
    mag_targ = df[mag_column_name].values
    ra_star = df[offset_ra_column_name].values
    dec_star = df[offset_dec_column_name].values
    mag_star = df[offset_mag_column_name].values

    name_targ = coord_to_name(ra_targ, dec_targ)
    rahms_targ, decdms_targ = degree_to_hms(ra_targ, dec_targ)
    rahms_star, decdms_star = degree_to_hms(ra_star, dec_star)
    target_coords = SkyCoord(ra=ra_targ, dec=dec_targ, unit=(u.deg, u.deg), frame='icrs')
    offset_coords = SkyCoord(ra=ra_star, dec=dec_star, unit=(u.deg, u.deg), frame='icrs')

    # Calculate position angles and separations (East of North) if they were not previously computed
    if pos_angle_column_name == None:
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg).value
        separation = offset_coords.separation(target_coords).to(u.arcsecond).value
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        dra = dra.to(u.arcsec).value
        ddec = ddec.to(u.arcsec).value
        df['pos_angle'] = pos_angles
        df['separation'] = separation
        pos_angle_column_name = 'pos_angle'
        df.to_csv(table)
    else:
        pos_angles = df[pos_angle_column_name].values
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        dra = dra.to(u.arcsec).value
        ddec = ddec.to(u.arcsec).value

    # Create the starlis for LRIS
    f = open(output_folder+'/'+output_starlist+'.txt', "w")
    for i in range(len(name_targ)):
        if pos_angles[i] > 180: pos_angles[i] -= 360

        print('{:} {:} {:} 2000.00 rotdest={:0.2f} rotmode=PA vmag={:0.1f}'.format(
            name_targ[i], rahms_targ[i].replace(':', ' '), decdms_targ[i].replace(':', ' '), pos_angles[i],
            mag_targ[i]), file=f)
        print('{:}_OFF  {:} {:} 2000.00 rotdest={:0.2f} rotmode=PA vmag={:0.1f} raoffset={:0.2f} '
              'decoffset={:0.2f}'.format(name_targ[i], rahms_star[i].replace(':', ' '),
                decdms_star[i].replace(':', ' '), pos_angles[i], mag_star[i], dra[i], ddec[i]), file=f)
    f.close()
    print('Starlist generated for observing with LRIS')

    # Download the cutouts
    survey_dict = [
        {'survey': survey, 'bands': [band],
         'fov': fov}
    ]

    whim.get_images(ra_targ,
                    dec_targ,
                    'cutouts',
                    survey_dict,
                    n_jobs=n_jobs)

    # Create the finding charts
    whfc.make_finding_charts(df, ra_column_name, dec_column_name,
                             target_column_name, survey, band,
                             aperture, fov, image_folder_path,
                             offset_table=df,
                             offset_id=0,
                             offset_focus=False,
                             offset_ra_column_name=offset_ra_column_name,
                             offset_dec_column_name=offset_dec_column_name,
                             offset_id_column_name=target_column_name,
                             pos_angle_column_name=pos_angle_column_name,
                             offset_mag_column_name=offset_mag_column_name,
                             format='png',
                             auto_download=False)