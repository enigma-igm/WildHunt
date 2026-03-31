#!/usr/bin/env python
"""

Main module for generating the files required to perform observations at the
telescope.

"""
import os

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from wildhunt.utilities.general_utils import coord_to_name, degree_to_hms


def instrument_observations_setup(telescope, instrument, table, ra_column_name, dec_column_name, target_column_name,
                                  mag_column_name, offset_ra_column_name, offset_dec_column_name,
                                  offset_mag_column_name, output_starlist, pos_angle_column_name = None,
                                  survey = 'DELSDR9', band = 'r', aperture = 2, fov = 120,
                                  image_folder_path = './cutouts', n_jobs = 1):
    """ Calls the different functions that generate the required files (finding charts, starlist, OBs...) to observe
    with a specific telescope
        :param instrument: string
            The telescope that is used in the observation
        :param instrument: string
            The instrument that is used in the observation
        :param table: string
            Name of the table catalog
        :param ra_column_name: string
            Column name for the right ascension of the targets in decimal degrees
        :param dec_column_name: string
            Column name for the declination of the targets in decimal degrees
        :param mag_column_name: string
            Column name for the magnitude of the targets
        :param offset_ra_column_name: string
            Offset star dataframe right ascension column name
        :param offset_dec_column_name: string
            Offset star dataframe declination column name
        :param offset_mag_column_name: string
            Offset star dataframe magnitude column name
        :param output_starlist: string
            Name of the output starlist
        :param pos_angle_column_name: string
            Column name of the position angle between the target (at the center) and the star in degrees
        :param survey: string
            Survey name
        :param band: string
            Passband name
        :param aperture: float
            Size of the plotted aperture in arcseconds
        :param fov: float
            Field of view in arcseconds
        :param image_folder_path: string
            Path to where the image will be stored
        :param n_jobs: int
            Number of jobs to perform parallel download of the images for the FCs
        """
    try:
        if telescope == 'keck':
            df = keck(instrument, table, ra_column_name, dec_column_name, mag_column_name,
                  offset_ra_column_name, offset_dec_column_name, offset_mag_column_name, output_starlist,
                  pos_angle_column_name)

        elif telescope == 'vlt':
            df = vlt(instrument, table, ra_column_name, dec_column_name, mag_column_name,
                     offset_ra_column_name, offset_dec_column_name, offset_mag_column_name,
                     pos_angle_column_name, band)

        # if genertate_fcs == True:
        #     finding_charts_generator(df, ra_column_name, dec_column_name, target_column_name, offset_ra_column_name,
        #                             offset_dec_column_name, offset_mag_column_name, pos_angle_column_name,
        #                             survey, band, aperture, fov, image_folder_path, n_jobs)

    except:
        print('ERROR: instrument not yet implemented')

# Below are defined the functions to prepare the observations for the different instruments

def keck(instrument, table, ra_column_name, dec_column_name, mag_column_name,
              offset_ra_column_name, offset_dec_column_name, offset_mag_column_name, output_starlist,
              pos_angle_column_name = None):
    """ Function that generates the starlist to observe with Keck
            :param instrument: string
                Name of the instrument used from Keck
            :param table: string
                Name of the table catalog
            :param ra_column_name: string
                Column name for the right ascension of the targets in decimal degrees
            :param dec_column_name: string
                Column name for the declination of the targets in decimal degrees
            :param mag_column_name: string
                Column name for the magnitude of the targets
            :param offset_ra_column_name: string
                Offset star dataframe right ascension column name
            :param offset_dec_column_name: string
                Offset star dataframe declination column name
            :param offset_mag_column_name: string
                Offset star dataframe magnitude column name
            :param output_starlist: string
                Name of the output starlist
            :param pos_angle_column_name: string
                Column name of the position angle between the target (at the center) and the star in degrees
            """

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
        df.to_csv(table, index = False)
    else:
        pos_angles = df[pos_angle_column_name].values
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        dra = dra.to(u.arcsec).value
        ddec = ddec.to(u.arcsec).value

    # Create the starlist for Keck
    if instrument == 'mosfire':
        pos_angles -= 4

    f = open(output_folder+'/'+output_starlist+'.txt', "w")
    for i in range(len(name_targ)):
        if pos_angles[i] > 180: pos_angles[i] -= 360

        ra_name = name_targ[i][:5]
        dec_name = name_targ[i][10:15]
        name = ra_name + dec_name

        print('{:} {:} {:} 2000.00 rotdest={:0.2f} rotmode=PA vmag={:0.1f}'.format(
            name + ' ' * 5, rahms_targ[i].replace(':', ' '), decdms_targ[i].replace(':', ' '), pos_angles[i],
            mag_targ[i]), file=f)
        print('{:}_OFF  {:} {:} 2000.00 rotdest={:0.2f} rotmode=PA vmag={:0.1f} raoffset={:0.2f} '
              'decoffset={:0.2f}'.format(name, rahms_star[i].replace(':', ' '),
                decdms_star[i].replace(':', ' '), pos_angles[i], mag_star[i], dra[i], ddec[i]), file=f)
    f.close()
    print('Starlist {}.txt generated for observing with Keck/{}'.format(output_starlist, instrument))

    return df

def vlt(instrument, table, ra_column_name, dec_column_name, mag_column_name,
              offset_ra_column_name, offset_dec_column_name, offset_mag_column_name,
              pos_angle_column_name = None, band = 'r'):
    """ Function that opens an existing OB and modify it to add the important info
            :param instrument: string
                Name of the instrument used from Keck
            :param table: string
                Name of the table catalog
            :param ra_column_name: string
                Column name for the right ascension of the targets in decimal degrees
            :param dec_column_name: string
                Column name for the declination of the targets in decimal degrees
            :param mag_column_name: string
                Column name for the magnitude of the targets
            :param offset_ra_column_name: string
                Offset star dataframe right ascension column name
            :param offset_dec_column_name: string
                Offset star dataframe declination column name
            :param offset_mag_column_name: string
                Offset star dataframe magnitude column name
            :param pos_angle_column_name: string
                Column name of the position angle between the target (at the center) and the star in degrees
            :param band: string
                Band from where mag_column_name was taken
            """

    # Open the candidates catalog
    df = pd.read_csv(table)

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
        df['pos_angle'] = pos_angles
        df['separation'] = separation
        df.to_csv(table, index = False)
    else:
        pos_angles = df[pos_angle_column_name].values

    # Create the directory to save the OBs
    path = os.getcwd()
    if os.path.isdir('{}_OBs'.format(instrument)):
        print("Directory {}_OBs already exists".format(instrument))
    else:
        print("Creation of the directory: {}_OBs".format(instrument))
        os.mkdir(path + '/{}_OBs'.format(instrument))

    for i, angle in enumerate(pos_angles):

        if angle >= 180:
            pos_angles[i] = angle - 360

        with open('ob_example.obx', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('Longslit_observations', name_targ[i])
        filedata = filedata.replace('Name_target', name_targ[i])
        filedata = filedata.replace('RA_HMS', rahms_star[i])
        filedata = filedata.replace('DEC_DMS', decdms_star[i])
        filedata = filedata.replace('mag_band', band + '=' + str(round(mag_star[i], 2)))
        filedata = filedata.replace('rot_angle', str(pos_angles[i]))

        # Write the file out again
        with open('{}_OBs/{}.obx'.format(instrument, name_targ[i]), 'w') as file:
            file.write(filedata)

    return df
