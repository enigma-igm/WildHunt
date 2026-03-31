#!/usr/bin/env python

import os
import string
from urllib.request import urlopen

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.irsa import Irsa
from astroquery.ukidss import Ukidss
from astroquery.vizier import Vizier
from astroquery.vsa import Vsa

from wildhunt import pypmsgs

msgs = pypmsgs.Messages()


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
                    'ukidssdr11las': {'service': 'UKIDSS', 'catalog': 'LAS',
                                      'ra': 'ra', 'dec': 'dec',
                                      'data_release': 'UKIDSSDR11PLUS',
                                      'mag_name': 'UKIDSS_J',
                                      'mag': 'jAperMag3', 'distance':
                                      'distance'},
                    # new, needs to be tested!
                    'vikingdr5': {'service': 'VSA', 'catalog': 'VIKING',
                                  'ra': 'ra', 'dec': 'dec',
                                  'data_release': 'VIKINGDR5', 'mag_name':
                                      'VHS_J',
                                  'mag': 'jAperMag3', 'distance': 'distance'}
                    # 'uhds': {'service': }
                  }

example_datalab_dict = {'table': 'ls_dr9.tractor',
                        'ra': 'ra',
                        'dec': 'dec',
                        'mag': 'mag_z',
                        'mag_name': 'lsdr9_z'}

# ------------------------------------------------------------------------------
#  Query functions for different services
# ------------------------------------------------------------------------------


def query_region_astroquery(ra, dec, match_distance, service, catalog,
                            data_release=None):
    """ Returns the catalog data of sources within a given radius of a defined
    position using astroquery.

    :param ra: Right ascension in decimal degrees.
    :type ra: float
    :param dec: Declination in decimal degrees.
    :type dec: float
    :param match_distance: Cone search radius in arcseconds
    :type match_distance: float
    :param service: Astroquery class used to query the catalog of choice
    :type service: string
    :param catalog: Astroquery catalog to query
    :type catalog: string
    :param data_release:  If needed by astroquery the specified data release
        (e.g. needed for VSA)
    :type data_release: string
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    if service == 'VIZIER':
        result = Vizier.query_region(target_coord,
                                     radius=match_distance * u.arcsecond,
                                     catalog=catalog, spatial='Cone')
        result = result[0]

    elif service == 'IRSA':
        result = Irsa.query_region(target_coord,
                                   radius=match_distance * u.arcsecond,
                                   catalog=catalog, spatial='Cone')
    elif service == 'VSA':
        result = Vsa.query_region(target_coord,
                                  radius=match_distance * u.arcsecond,
                                  programme_id=catalog, database=data_release)
    elif service == 'UKIDSS':
        result = Ukidss.query_region(target_coord,
                                     radius=match_distance * u.arcsecond,
                                     programme_id=catalog,
                                     database=data_release)
    else:
        raise KeyError('Astroquery service not recognized. Implemented '
                       'services include: Vizier, Irsa, VSA, Ukidss')

    return result.to_pandas()


def get_astroquery_offset(target_name, target_ra, target_dec, match_distance,
                          catalog,  quality_query=None, n=3,
                          minimum_distance=3):
    """Return the nth nearest offset stars specified by the quality criteria
    around a given target using astroquery.

    :param target_name:  Identifier for the target
    :type target_name: string
    :param target_ra: Target right ascension in decimal degrees.
    :type target_ra: float
    :param target_dec: Target Declination in decimal degrees.
    :type target_dec: float
    :param match_distance: Maximum search radius in arcseconds
    :type match_distance: float
    :param catalog: Catalog (and data release) to retrieve the offset star
     data from. See astroquery_dict for implemented catalogs.
    :type: string
    :param quality_query:  A string written in pandas query syntax to apply
        quality criteria on potential offset stars around the target.
    :type quality_query: string
    :param n: Number of offset stars to retrieve. (Maximum: n=26)
    :type n: int
    :param minimum_distance: Minimum distance to the target in arcsec
    :type minimum_distance: float
    :return: Returns the dataframe with the retrieved offset stars for the
     given target.
    :rtype: pandas.core.frame.DataFrame
    """

    # Restrict the number of offset stars to be returned to n=26
    if n > 26:
        msgs.info('Maximum number of offset stars is 26. Setting n to 26.')
        n = 26

    service = astroquery_dict[catalog]['service']
    cat = astroquery_dict[catalog]['catalog']
    ra = astroquery_dict[catalog]['ra']
    dec = astroquery_dict[catalog]['dec']
    mag = astroquery_dict[catalog]['mag']
    mag_name = astroquery_dict[catalog]['mag_name']
    distance = astroquery_dict[catalog]['distance']
    dr = astroquery_dict[catalog]['data_release']

    df = query_region_astroquery(target_ra, target_dec, match_distance,
                                 service, cat, dr).copy()

    letters = string.ascii_uppercase

    if quality_query is not None:
        df.query(quality_query, inplace=True)

    # Querying for a minimum distance (converted to arcminutes here)
    df.query('{} > {}'.format(distance, minimum_distance/60.), inplace=True)

    if df.shape[0] > 0:
        # Sort DataFrame by match distance

        df.sort_values(distance, ascending=True, inplace=True)
        # Keep only the first three n entries
        offset_df = df[:n].copy()

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]
        for jdx, idx in enumerate(offset_df.index):
            letter = letters[jdx]

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
        msgs.warn("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()


def get_datalab_offset(target_name, target_ra, target_dec, radius,
                       datalab_dict, columns=None, where=None, n=3,
                       minimum_distance=3, verbosity=0):
    """Return the nth nearest offset stars specified by the quality criteria
    around a given target using the NOAO datalab.

    :param target_name:  Identifier for the target
    :type target_name: string
    :param target_ra: Target right ascension in decimal degrees.
    :type target_ra: float
    :param target_dec: Target Declination in decimal degrees.
    :type target_dec: float
    :param radius: Maximum search radius in arcseconds
    :type radius: float
    :param datalab_dict: Survey dictionary for the datalab query.
    :type datalab_dict: dict
    :param columns: Columns names returned from datalab table.
    :type columns: list or None
    :param where: A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :type where: string
    :param n: Number of offset stars to retrieve. (Maximum: n=5)
    :type n: int
    :param minimum_distance: Minimum distance to the target in arcsec
    :type minimum_distance: float
    :param verbosity: Verbosity > 0 will print verbose statements during the
        execution.
    :type verbosity: int
    :return:  Returns the dataframe with the retrieved offset stars for the
        given target.
    :rtype: pandas.core.frame.DataFrame
    """

    # Restrict the number of offset stars to be returned to n=26
    if n > 26:
        msgs.info('Maximum number of offset stars is 26. Setting n to 26.')
        n = 26

    df = query_region_datalab(target_ra, target_dec, radius,
                              datalab_dict=datalab_dict,
                              columns=columns, where=where,
                              verbosity=verbosity,
                              minimum_distance=minimum_distance)

    ra = datalab_dict['ra']
    dec = datalab_dict['dec']
    mag = datalab_dict['mag']
    mag_name = datalab_dict['mag_name']

    letters = string.ascii_uppercase

    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df.iloc[:n, :].copy()

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]

        offset_df['offset_name'] = pd.Series(dtype=pd.StringDtype())
        offset_df['offset_shortname'] = pd.Series(dtype=pd.StringDtype())

        for jdx, idx in enumerate(offset_df.index):
            letter = letters[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' + \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter

            offset_df.loc[:, mag_name] = df[mag]

        # ToDo: GET THIS INTO A SEPARATE FUNCTION
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
        msgs.warn("Offset star for {} not found.".format(target_name))

        return pd.DataFrame()


def query_region_datalab(ra, dec, radius, datalab_dict,
                         columns=None, where=None,
                         minimum_distance=3, verbosity=0):
    """ Returns the catalog data of sources within a given radius of a defined
    position using the NOIRLAB astro data lab.

    :param ra: Right ascension in decimal degrees.
    :type ra: float
    :param dec: Declination in decimal degrees.
    :type dec: float
    :param radius: Cone search radius in arcseconds
    :type radius: float
    :type datalab_dict: dict
    :param columns: Columns names returned from datalab table.
    :type columns: list
    :param where: A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :param minimum_distance: Minimum distance to the target in arcsec
    :type minimum_distance: float
    :param verbosity: Verbosity > 0 will print verbose statements during the
        execution.
    :type verbosity: int
    :return: Returns the dataframe with the returned matches
    :rtype:  pandas.core.frame.DataFrame
    """
    from dl import queryClient as qc
    
    radius_deg = radius / 3600.

    minimum_distance = minimum_distance / 3600.

    table = datalab_dict['table']

    # Build SQL query
    if columns is not None:
        sql_query = 'SELECT {} '.format(columns)
    else:
        sql_query = 'SELECT * '

    sql_query += ', q3c_dist(ra, dec, {}, {}) as distance '.format(ra, dec)

    sql_query += 'FROM {} WHERE '.format(table)

    sql_query += 'q3c_radial_query(ra, dec, {}, {}, {}) '.format(ra, dec,
                                                                 radius_deg)
    if where is not None:
        sql_query += 'AND {}'.format(where)

    # Query DataLab and write result to temporary file
    if verbosity > 1:
        msgs.info("SQL QUERY: {}".format(sql_query))

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

    :param ra: Right ascension in decimal degrees.
    :type ra: float
    :param dec: Declination in decimal degrees.
    :type dec: float
    :param radius: Cone search radius in arcseconds
    :type radius: float
    :param survey: Survey keyword for the PanSTARRS MAST query.
    :type survey: string
    :param catalog: Catalog keyword for the PanSTARRS MAST query.
    :type catalog: string
    :param add_criteria:  A string with conditions to apply additional
        quality criteria on potential offset stars around the target.
    :type add_criteria: string or None
    :param verbosity: Verbosity > 0 will print verbose statements during the
        execution.
    :type verbosity: int
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    urlbase = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/'

    if add_criteria is None:
        url = urlbase + \
              '{}/{}.csv?ra={}&dec={}&radius={}&format=csv'.format(survey,
                                                                catalog,
                                                               ra,
                                                               dec,
                                                               radius)
    else:
        # FIXME: This might be a bug, ask JT
        url = urlbase + '{}/{}?ra={}&dec={}&radius={}&' + \
              add_criteria + 'format=csv'.format()
    if verbosity > 0:
        msgs.info('Opening {}'.format(url))

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
    """Return the nth nearest offset stars specified by the quality criteria
    around a given target using the MAST website for PanSTARRS.

    It will always retrieve the z-band magnitude for the offset star. This is
    hardcoded. Depending on the catalog it will be the mean of stack magnitude.

    :param target_name:  Identifier for the target
    :type target_name: string
    :param target_ra: Target right ascension in decimal degrees.
    :type target_ra: float
    :param target_dec: Target Declination in decimal degrees.
    :type target_dec: float
    :param radius: Cone search radius in arcseconds
    :type radius: float
    :param catalog:  Catalog to retrieve the offset star data from.
        (e.g. 'mean', 'stack')
    :type catalog: string
    :param data_release: The specific PanSTARRS data release
    :type data_release: string
    :param quality_query:  A string written in pandas query syntax to apply
        quality criteria on potential offset stars around the target.
    :type quality_query: string
    :param n: Number of offset stars to retrieve. (Maximum: n=5)
    :type n: int
    :param verbosity: Verbosity > 0 will print verbose statements during the
        execution.
    :type verbosity: int
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    # Restrict the number of offset stars to be returned to n=26
    if n > 26:
        msgs.info('Maximum number of offset stars is 26. Setting n to 26.')
        n = 26

    # Convert radius in degrees
    radius_degree = radius / 3600.

    letters = string.ascii_uppercase

    if verbosity > 0:
        msgs.info('Querying PS1 Archive ({},{}) for {}'.format(catalog,
                                                           data_release,
                                                           target_name))
    # Query the PanStarrs 1 archive
    df = query_region_ps1(target_ra, target_dec, radius_degree,
                          survey=data_release,
                          catalog=catalog, add_criteria=None,
                          verbosity=verbosity)

    if df is None:
        msgs.warn("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()

    # Drop duplicated targets
    df.drop_duplicates(subset='objName', inplace=True)

    # Apply quality criteria query
    if quality_query is not None:
        df.query(quality_query, inplace=True)
    if df.shape[0] > 0:

        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n].copy()

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = offset_df.loc[:, 'raMean']
        offset_df.loc[:, 'offset_dec'] = offset_df.loc[:, 'decMean']
        for jdx, idx in enumerate(offset_df.index):

            letter = letters[jdx]

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
                'Catalog value not understood ["mean","stack"] :{}'.format(
                    catalog))

        target_coords = SkyCoord(ra=target_ra, dec=target_dec,
                                 unit=(u.deg, u.deg), frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec,
                                 unit=(u.deg, u.deg), frame='icrs')
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

        return offset_df[['target_name', 'target_ra', 'target_dec',
                          'offset_name', 'offset_shortname', 'offset_ra',
                          'offset_dec', mag, 'separation', 'pos_angle',
                          'dra_offset', 'ddec_offset']]
    else:
        msgs.warn("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()
