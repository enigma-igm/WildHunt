#!/usr/bin/env python

import os
import pandas as pd

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits

from urllib.request import urlopen  # python3
from urllib.error import HTTPError
from http.client import IncompleteRead

from astroquery.vizier import Vizier
from astroquery.irsa import Irsa
from astroquery.vsa import Vsa
from astroquery.ukidss import Ukidss

from dl import queryClient as qc

# def get_offset_stars(target_df, ra_column_name, dec_column_name,
#                      offset_survey='UKIDSSDR11PLUSLAS', offset_columns,
#                      n_star=3):
#
#     if statements with
#
#     if survey_name[:3] in ['VHS', 'VVV', 'VMC', 'VIK', 'VID', 'UKI']
# FOCUS ON GETTING UKIDSS and LEGACY SURVEY RUNNING


# def crossmatch_with_catalog(df, ra_column_name, dec_column_name, survey_name,
#                             mode='closest', columns_to_append=list of column
#                             names)p

# ------------------------------------------------------------------------------
#  Supported surveys, data releases, bands
# ------------------------------------------------------------------------------

astroquery_dict = {
                    'tmass': {'service': 'irsa', 'catalog': 'fp_psc',
                              'ra': 'ra', 'dec': 'dec', 'mag_name':
                              'TMASS_J', 'mag': 'j_m', 'distance':
                              'dist', 'data_release': None},
                    'nomad': {'service': 'vizier', 'catalog': 'NOMAD',
                              'ra': 'RAJ2000', 'dec': 'DECJ2000',
                              'mag_name': 'R', 'mag': 'Rmag', 'distance':
                              'distance', 'data_release': None},
                    'vhsdr6': {'service': 'vsa', 'catalog': 'VHS',
                               'ra': 'ra', 'dec': 'dec',
                               'data_release': 'VHSDR6', 'mag_name': 'VHS_J',
                               'mag': 'jAperMag3', 'distance': 'distance'},
                    'ukidssdr11': {'service': 'UKIDSS', 'catalog': 'LAS',
                               'ra': 'ra', 'dec': 'dec',
                               'data_release': 'UKIDSSDR11PLUS', 'mag_name':
                                       'UKIDSS_J',
                               'mag': 'jAperMag3', 'distance': 'distance'},
                    # new, needs to be tested!
                    'vikingdr5': {'service': 'vsa', 'catalog': 'VIKING',
                               'ra': 'ra', 'dec': 'dec',
                               'data_release': 'VIKINGDR5', 'mag_name': 'VHS_J',
                               'mag': 'jAperMag3', 'distance': 'distance'}
                  }

example_datalab_dict = {'survey': 'ls_dr9',
                        'table': 'tractor',
                        'ra': 'ra',
                        'dec': 'dec',
                        'mag': 'mag_z',
                        'mag_name': 'lsdr9_z'}

# ------------------------------------------------------------------------------
#  Supported surveys, data releases, bands
# ------------------------------------------------------------------------------


def query_region_astroquery(ra, dec, radius, service, catalog,
                            data_release=None):
    """ Returns the catalog data of sources within a given radius of a defined
    position using astroquery.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in arcseconds
    :param service: string
        Astroquery class used to query the catalog of choice
    :param catalog: string
        Catalog to query
    :param data_release:
        If needed by astroquery the specified data release (e.g. needed for VSA)
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    if service == 'VIZIER':
        result = Vizier.query_region(target_coord, radius=radius * u.arcsecond,
                                     catalog=catalog, spatial='Cone')
        result = result[0]

    elif service == 'IRSA':
        result = Irsa.query_region(target_coord, radius=radius * u.arcsecond,
                                   catalog=catalog, spatial='Cone')
    elif service == 'VSA':
        result = Vsa.query_region(target_coord, radius=radius * u.arcsecond,
                                   programme_id=catalog, database=data_release)
    elif service == 'UKIDSS':
        result = Ukidss.query_region(target_coord, radius=radius * u.arcsecond,
                                   programme_id=catalog, database=data_release)
    else:
        raise KeyError('Astroquery class not recognized. Implemented classes '
                       'are: Vizier, Irsa, VSA, Ukidss')

    return result.to_pandas()


def get_astroquery_offset(target_name, target_ra, target_dec, radius, catalog,
                          quality_query=None, n=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using astroquery.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog (and data release) to retrieve the offset star data from. See
        astroquery_dict for implemented catalogs.
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """


    service = astroquery_dict[catalog]['service']
    cat = astroquery_dict[catalog]['catalog']
    ra = astroquery_dict[catalog]['ra']
    dec = astroquery_dict[catalog]['dec']
    mag = astroquery_dict[catalog]['mag']
    mag_name = astroquery_dict[catalog]['mag_name']
    distance = astroquery_dict[catalog]['distance']
    dr = astroquery_dict[catalog]['data_release']

    df = query_region_astroquery(target_ra, target_dec, radius, service, cat,
                                 dr).copy()

    if quality_query is not None:
        df.query(quality_query, inplace=True)

    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values(distance, ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n]


        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]
        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' +  \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter


            offset_df.loc[:, mag_name] = df[mag]

        # GET THIS INTO A SEPARATE FUNCTION
        target_coords = SkyCoord(ra=target_ra, dec=target_dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')


        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)

        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[
            ['target_name', 'target_ra', 'target_dec', 'offset_name',
             'offset_shortname', 'offset_ra', 'offset_dec',
             mag, 'separation', 'pos_angle', 'dra_offset',
             'ddec_offset']]
    else:
        print("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()



def get_offset_stars_astroquery(df, target_name_column, target_ra_column,
                     target_dec_column, radius, catalog='tmass', n=3,
                                quality_query=None, verbosity=0):
    """Get offset stars for all targets in the input DataFrame using astroquery.


    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog (and data release) to retrieve the offset star data from. See
        astroquery_dict for implemented catalogs.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """

    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]


        temp_df = get_astroquery_offset(target_name, target_ra, target_dec, radius, catalog,
                          quality_query=quality_query, n=n, verbosity=verbosity)


        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    return offset_df


def get_offset_stars_datalab(df, target_name_column, target_ra_column,
                     target_dec_column, radius, datalab_dict,
                             n=3, where=None, minimum_distance=
                             3./60, verbosity=0):
    """Get offset stars for all targets in the input DataFrame using the
    NOAO datalab.

    Example of a datalab_dict

    dict = {'survey': 'ls_dr9',
            'table': 'tractor',
            'ra': 'ra',
            'dec': 'dec',
            'mag': 'mag_z',
            'mag_name': 'lsdr9_z'}

    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """

    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]

        temp_df = get_datalab_offset(target_name, target_ra, target_dec, radius,
                                     datalab_dict, columns=None,
                                     where=where, n=n,
                                     minimum_distance=minimum_distance,
                                     verbosity=verbosity)

        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    os.remove('temp_offset_df.csv')

    return offset_df


def get_datalab_offset(target_name, target_ra, target_dec, radius,
                       datalab_dict, columns=None, where=None, n=3,
                       minimum_distance=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using the NOAO datalab.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """

    df = query_region_datalab(target_ra, target_dec, radius,
                              datalab_dict=datalab_dict,
                              columns=columns, where=where,
                              verbosity=verbosity,
                              minimum_distance=minimum_distance)

    ra = datalab_dict['ra']
    dec = datalab_dict['dec']
    mag = datalab_dict['mag']
    mag_name = datalab_dict['mag_name']

    # distance column is in arcminutes!!

    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df.iloc[:n, :]

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]

        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' + \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter

            offset_df.loc[:, mag_name] = df[mag]

        # GET THIS INTO A SEPARATE FUNCTION
        target_coords = SkyCoord(ra=target_ra, dec=target_dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')

        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        # UNTIL HERE

        if verbosity > 1:
            print('Offset delta ra: {}'.format(dra))
            print('Offset delta dec: {}'.format(ddec))
            print('Offset separation: {}'.format(separations))
            print('Offset position angle: {}'.format(pos_angles))

        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[
            ['target_name', 'target_ra', 'target_dec', 'offset_name',
             'offset_shortname', 'offset_ra', 'offset_dec',
             mag, 'separation', 'pos_angle', 'dra_offset',
             'ddec_offset']]

    else:
        print("Offset star for {} not found.".format(target_name))

        return pd.DataFrame()


def query_region_datalab(ra, dec, radius, datalab_dict,
                         columns=None, where=None,
                         minimum_distance=3, verbosity=0):
    """ Returns the catalog data of sources within a given radius of a defined
    position using the NOAO datalab.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param columns:
        The names of the columns that should be returned.
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    radius_deg = radius / 3600.

    minimum_distance = minimum_distance / 3600.

    survey = datalab_dict['survey']
    table = datalab_dict['table']

    # Build SQL query
    if columns is not None:
        sql_query = 'SELECT {} '.format(columns)
    else:
        sql_query = 'SELECT * '.format(survey, table)

    sql_query += ', q3c_dist(ra, dec, {}, {}) as distance '.format(ra, dec)

    sql_query += 'FROM {}.{} WHERE '.format(survey, table)

    sql_query += 'q3c_radial_query(ra, dec, {}, {}, {}) '.format(ra, dec,
                                                                radius_deg)
    if where is not None:
        sql_query += 'AND {}'.format(where)

    # Query DataLab and write result to temporary file
    if verbosity > 0:
        print("SQL QUERY: {}".format(sql_query))

    result = qc.query(sql=sql_query)

    f = open('temp.csv', 'w+')
    f.write(result)
    f.close()

    # Open temporary file in a dataframe and delete the temporary file
    df = pd.read_csv('temp.csv')

    df.query('distance > {}'.format(minimum_distance), inplace=True)

    os.remove('temp.csv')

    return df


def query_region_ps1(ra, dec, radius, survey='dr2', catalog='mean',
                     add_criteria=None, verbosity=0):
    """ Returns the catalog data of sources within a given radius of a defined
    position using the MAST website.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in degrees
    :param survey: string
        Survey keyword for the PanSTARRS MAST query.
    :param catalog: string
        Catalog keyword for the PanSTARRS MAST query.
    :param columns:
        The names of the columns that should be returned.
    :param  add_criteria: string
        A string with conditions to apply additional quality criteria on
        potential offset stars around the target.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """
    urlbase = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/'

    if add_criteria is None:
        url = urlbase + \
              '{}/{}?ra={}&dec={}&radius={}&format=csv'.format(survey, catalog,
                                                               ra,
                                                               dec,
                                                               radius)
    else:
        url = urlbase + '{}/{}?ra={}&dec={}&radius={}&' + \
              add_criteria + 'format=csv'.format(survey, catalog, ra, dec,
                                                 radius)
    if verbosity>0:
        print('Opening {}'.format(url))

    response = urlopen(url)
    check_ok = response.msg == 'OK'

    empty = response.read() == b''

    if check_ok and not empty:
        df = pd.read_csv(url)
        return df

    elif check_ok and empty:
        if verbosity > 0:
            print('Returned file is empty. No matches found.')
        return None

    else:
        raise ValueError('Could not retrieve PS1 data.')


def get_ps1_offset_star(target_name, target_ra, target_dec, radius=300,
                        catalog='mean', data_release='dr2',
                        quality_query=None, n=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using the MAST website for PanSTARRS.

    It will always retrieve the z-band magnitude for the offset star. This is
    hardcoded. Depending on the catalog it will be the mean of stack magnitude.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog to retrieve the offset star data from. (e.g. 'mean', 'stack')
    :param data_release: string
        The specific PanSTARRS data release
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """

    # Convert radius in degrees
    radius_degree = radius / 3600.

    if verbosity>1:
        print('Querying PS1 Archive ({},{}) for {}'.format(catalog,
                                                           data_release,
                                                           target_name))
    # Query the PanStarrs 1 archive
    df = query_region_ps1(target_ra, target_dec, radius_degree,
                          survey=data_release,
                          catalog=catalog, add_criteria=None,
                          verbosity=verbosity)

    # Drop duplicated targets
    df.drop_duplicates(subset='objName', inplace=True)
    # Apply quality criteria query
    if quality_query is not None:
        df.query(quality_query, inplace=True)
    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n]

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df.raMean
        offset_df.loc[:, 'offset_dec'] = df.decMean
        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' + \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter

        if catalog == 'mean':
            mag = 'ps1_' + data_release + '_mean_psfmag_y'
            offset_df.loc[:, mag] = df.yMeanPSFMag
        elif catalog == 'stack':
            mag = 'ps1_' + data_release + '_stack_psfmag_y'
            offset_df.loc[:, mag] = df.yPSFMag
        else:
            raise ValueError(
                'Catalog value not understood ["mean","stack"] :{}'.format(catalog))

        target_coords = SkyCoord(ra=target_ra, dec=target_dec, unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')
        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)

        if verbosity > 1:
            print('Offset delta ra: {}'.format(dra))
            print('Offset delta dec: {}'.format(ddec))
            print('Offset separation: {}'.format(separations))
            print('Offset position angle: {}'.format(pos_angles))


        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[['target_name', 'target_ra', 'target_dec', 'offset_name',
                          'offset_shortname', 'offset_ra', 'offset_dec',
                          mag, 'separation', 'pos_angle', 'dra_offset',
                          'ddec_offset']]
    else:
        print("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()


def get_offset_stars_ps1(df, target_name_column, target_ra_column,
                     target_dec_column, radius, data_release='dr2',
                     catalog='mean', quality_query=None, n=3, verbosity=0):
    """Get offset stars for all targets in the input DataFrame for PanSTARRS
    using the MAST website.

    Currently this runs slowly as it queries the PanSTARRS 1 archive for each
    object. But it runs!

    It will always retrieve the z-band magnitude for the offset star. This is
    hardcoded in get_ps1_offset_star(). Depending on the catalog it will be
    the mean of stack magnitude.


    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
       :param catalog: string
        Catalog to retrieve the offset star data from. (e.g. 'mean', 'stack')
    :param data_release: string
        The specific PanSTARRS data release
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """
    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]


        temp_df = get_ps1_offset_star(target_name, target_ra, target_dec,
                                        radius=radius, catalog=catalog,
                                        data_release=data_release,
                                        quality_query=quality_query, n=n,
                                      verbosity=verbosity)


        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    return offset_df