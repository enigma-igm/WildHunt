#!/usr/bin/env python

import os
import glob
import getpass
import shutil
import numpy as np
import pandas as pd

import dask.dataframe as dd

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky

from dl import authClient as ac, queryClient as qc, storeClient as sc
from dl.helpers.utils import convert

from IPython import embed

from wildhunt import catalog_defaults as whcd
from wildhunt import catalog_queries as whcq
from wildhunt import pypmsgs
from wildhunt import utils
from wildhunt.surveys import panstarrs, vsa_wsa, legacysurvey
msgs = pypmsgs.Messages()

def retrieve_survey(survey_name, bands, fov, verbosity=1):
    """ Retrieve survey class according to the survey name.

    :param survey_name:
    :param bands:
    :param fov:
    :return:
    """

    survey = None

    if survey_name == 'PS1':
        survey = panstarrs.Panstarrs(bands, fov, verbosity=verbosity)

    if survey_name[:3] in ['VHS', 'VVV', 'VMC', 'VIK', 'VID', 'UKI', 'UHS']:
        survey = vsa_wsa.VsaWsa(bands, fov, survey_name, verbosity=verbosity)

    if survey_name[:4] == 'DELS':
        survey = legacysurvey.LegacySurvey(bands, fov, survey_name,
                                           verbosity=verbosity)

    if survey is None:
        print('ERROR')

    return survey

class Catalog(object):
    """ Catalog class to handle small to large operations on photometric
    catalogs.

    Catalogs are operated on internally as dask dataframes to guarantee
    memory and computation efficient catalog operations.

    The standard file format is parquet. As a default catalogs are stored
    in folders with N parquet files according to the number of dask
    dataframe partitions.

    """

    def __init__(self, name, ra_column_name, dec_column_name,
                 id_column_name, datapath=None, table_data=None,
                 clear_temp_dir=True, dtype=None, verbose=1):
        """ Initialize catalog class

        :param name: Name of the catalog (is used as an identified for
        saving the catalog or in cross-matches to the catalog)
        :type name: string
        :param ra_column_name: Name of the RA column
        :type ra_column_name: string
        :param dec_column_name: Name of the DEC column
        :type dec_column_name: string
        :param id_column_name: Name of the identifier column. The identifier
            should ideally be unqiue to the source.
        :type id_column_name: string
        :param datapath: Path to the catalog data. The data can be in a
            single file or in a folder.
        :type datapath: string
        :param table_data: Table data as a pandas dataframe.
        :type table_data: pandas.core.frame.DataFrame
        :param clear_temp_dir:  Erase the temporary directory after each
            operation in which it would be created.
        :type clear_temp_dir: bool
        :param dtype: List of data types for data input from csv files to
            allow for optimal storage.
        :type dtype: list(string)
        :param verbose: Verbosity level for functions.
        :type verbose: int
        """

        # Main attributes
        self.name = name  # Name of the catalog
        self.datapath = datapath  # Path to catalog data
        self.df = None  # Dask dataframe (populated internally)
        self.verbose = verbose  # Verbosity level
        self.dtype = dtype  # List of data types

        # Column name attributes
        self.id_colname = id_column_name  # Identifier column name
        self.ra_colname = ra_column_name  # RA column name
        self.dec_colname = dec_column_name  # DEC column name
        self.columns = [self.id_colname, self.ra_colname, self.dec_colname]

        # Attributes for file
        self.chunk = None  # Dataframe of the current batch
        self.temp_dir = 'wild_temp'  # Name of the temporary directory
        self.clear_temp_dir = clear_temp_dir  # Erase/Keep temporary
        # dictionary

        # Attributes for file partitioning
        self.partition_limit = 1073741824  # Maximum catalog size; 1GB
        self.partition_size = 104857600  # Maximim partition size; 100Mb

        # Instantiate dask dataframe
        if table_data is None and datapath is None:
            msgs.error('No dataframe or datapath specified.')
        elif table_data is None and datapath is not None:
            self.init_dask_dataframe()
        elif datapath is None and table_data is not None:
            self.df = dd.from_pandas(table_data, npartitions=1)

    def init_dask_dataframe(self):
        """Initialize dask dataframe.

        This function populates the df attribute by reading the data in
        datapath.

        If datapath is a file it opens the file and checks whether the file
        size exceeds self.partition_limit (default: 1GiB). If the file is
        smaller the entire file will be converted to a dask datafile with
        one partition. If the file is larger, it will be partitioned in
        partitions with a maximum size of self.partition_size (default:
        100Mib).

        If datapath is a folder, the function assumes that the folder is
        made up of one partition for every file in the folder.

        The default file format is parquet and saving the catalog will
        always be done in parquet files.

        For input the following file formats are additionally supported:
        csv, hdf5, fits (Table)

        :return:
        """

        msgs.info('Initializing catalog dataframe (dask dataframe)')

        file_suffix = self.datapath.split('.')[-1]
        partition_file = False
        read_fits = False
        filesize = 0
        df = None

        if os.path.isdir(self.datapath):
            files = glob.glob(os.path.join('{}/*'.format(self.datapath)))
            file_suffix = files[0].split('.')[-1]

        elif os.path.isfile(self.datapath):
            file_suffix = self.datapath.split('.')[-1]
            filesize = os.path.getsize(self.datapath)
            if filesize > self.partition_limit:

                partition_file = True
                readable_filesize = utils.sizeof_fmt(filesize)
                readable_partition_limit = utils.sizeof_fmt(
                    self.partition_limit)
                msgs.warn('You supplied a single file in excess'
                          ' of {}.'.format(readable_partition_limit))
                msgs.warn('File size: {}'.format(readable_filesize))
                msgs.warn('Your file will be converted into parquet '
                          'partitions.')

        if file_suffix == 'parquet':
            df = dd.read_parquet(self.datapath)
        elif file_suffix == 'csv':
            df = dd.read_csv(self.datapath, dtype=self.dtype)
        elif file_suffix == 'hdf5':
            df = dd.read_hdf(self.datapath, 'data')
        elif file_suffix == 'fits':
            msgs.warn('You provided data in the fits file format.')
            msgs.warn('This format is not natively supported in WildHunt.')
            msgs.warn('Please consider using parquet files in the future.')
            # Set fits flag for repartitioning
            read_fits = True
        else:
            msgs.error('Provided file format {} not supported'.format(
                file_suffix))

        if partition_file:
            # Repartition a single large input file
            msgs.info('Repartitioning large input file.')
            n_partitions = round(filesize/self.partition_size)
            msgs.info('Creating {} partitions.'.format(n_partitions))
            msgs.info('This may take a while.')
            # Repartition dataframe
            if read_fits:
                # Fits file conversion
                msgs.warn('Converting a large fits file.')
                table = Table.read(self.datapath)
                df = table.to_pandas()
                df = dd.from_pandas(df,
                                    npartitions=n_partitions)
            else:
                df = df.repartition(npartitions=n_partitions)

            # Save catalog structure
            self.df = df
            self.save_catalog()
            # Deleting class variable to free up space
            del self.df
            # Read in and reinitialize class variable
            self.df = dd.read_parquet(self.name)

        else:
            self.df = df

        msgs.info('Catalog dataframe initialized.')

    def save_catalog(self, filepath=None, file_format='parquet'):
        """ Save catalog to filepath

        :param filepath: Filepath to save the catalog
        :type filepath: string
        :param file_format: Format to save the catalog in
        :type file_format: string (default: parquet)
        :return: None
        """

        if filepath is None:
            filepath = self.name

        if file_format == 'parquet':
            self.df.to_parquet('{}'.format(filepath))
        elif file_format == 'csv':
            self.df.to_csv('{}'.format(filepath), index=False)
        elif file_format == 'hdf5':
            self.df.to_hdf('{}'.format(filepath), format='table')
        else:
            msgs.error('Provided file format {} not supported'.format(
                file_format))

        msgs.info('Saved catalog to {}'.format(filepath))

    def catalog_cross_match(self, match_catalog, match_distance,
                            columns='all', column_prefix=None):
        """ Merge catalog with external catalog (wildhunt.Catalog).

        :param match_catalog: Catalog to match the current catalog class to.
        :type match_catalog: wildhunt.Catalog
        :param match_distance: Match distance in arcseconds.
        :type match_distance: float
        :param columns: List of catalog columns to merge from input catalog.
        :type columns: list(string)
        :param column_prefix: Prefix to add to merged columns from input
            catalog.
        :type column_prefix: string
        :return: None
        """

        msgs.info('Cross-matching {} with {}'.format(self.name,
                                                     match_catalog.name))
        msgs.warn('The current implementation does not make optimal use '
                  'of dask dataframes.')
        msgs.warn('In consequence the full match dataframe is loaded in '
                  'memory.')

        # Create output directory
        output_dir = '{}_{}_merged'.format(self.name,
                                                  match_catalog.name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            msgs.info('Creating output directory')

        for idx, partition in enumerate(self.df.partitions):

            msgs.info('Matching partition {} out of {}'.format(
                idx+1, self.df.npartitions))

            # Create source pandas dataframe from partition
            source = partition.compute()
            # Create match pandas dataframe from partition
            match_df = match_catalog.df.compute()

            # Usual astropy procedure
            source_coord = SkyCoord(
                source[self.ra_colname].values * u.deg,
                source[self.dec_colname].values * u.deg)
            match_coord = SkyCoord(
                match_df[match_catalog.ra_colname].values * u.deg,
                match_df[match_catalog.dec_colname].values * u.deg)

            match_idx, separation, d3d = match_coordinates_sky(
                source_coord,
                match_coord)

            # Add match_index and match distance to df
            source['match_index'] = match_idx
            source['{}_distance'.format(column_prefix)] = separation.to(
                'arcsec').value

            # Generate columns for merge
            source.loc[:, '{}_match'.format(column_prefix)] = False
            cat_idx = source.query('{}_distance < {}'.format(column_prefix,
                                                              match_distance)).index
            source.loc[cat_idx, '{}_match'.format(column_prefix)] = True

            # Remove match_distance in cases the source exceeds the match distance
            cat_idx = source.query('{}_distance > {}'.format(column_prefix,
                                                              match_distance)).index
            source.loc[cat_idx, 'match_index'] = np.NaN
            source.loc[cat_idx, '{}_distance'.format(column_prefix)] = np.NaN

            # Merge catalog catalogs on merge index
            if columns == 'all':

                df_merged = source.merge(match_df,
                                           how='left',
                                           left_on='match_index',
                                           right_index=True)
            else:

                df_merged = source.merge(match_df[columns],
                                           how='left',
                                           left_on='match_index',
                                           right_index=True)

            filename = '{}_{}_merged/part.{}.parquet'.format(self.name,
                                                  match_catalog.name,
                                                  idx)
            df_merged.to_parquet(filename)

    def online_cross_match(self, survey='DELS', columns='default',
                           match_distance=3):
        """Positional cross-match to online catalog with a maximum match
        distance of match_distance (in arcseconds).

        Implemented online cross-matches:
             - Dark Energy Legacy Survey (DELS): DR9 Tractor
             table (ls_dr9.tractor)

        :param survey: Survey identifier
        :type survey: string
        :param columns: List of column names to merge from online catalog.
        :type columns: list
        :param match_distance: Match distance in arcseconds.
        :type match_distance: float
        :return:
        """

        msgs.info('Starting online cross match')
        msgs.info('Survey: {} '.format(survey))

        # Check if temporary folder exists
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
            msgs.info('Creating WildHunt temporary directory')

        # For datalab surveys log in to datalb
        if survey in ['DELS']:
            response = ac.whoAmI()
            if response == 'anonymous':
                msgs.info('Log in to NOIRLAB Astro Data Lab')
                token = ac.login(input('Enter user name (+ENTER): '),
                                 getpass.getpass('Enter password (+ENTER): '))
            msgs.info('Astro Data Lab USER: {}'.format(ac.whoAmI()))
            msgs.info('Astro Data Lab TABLES: {}'.format(qc.mydb_list()))

        # Serialized cross-match over partitions
        for idx, partition in enumerate(self.df.partitions):
            msgs.info('Beginning crossmatch on partition {}'.format(idx))
            self.chunk = partition.compute()[self.columns]

            # TODO a quick hack to fix later
            # I suggest to implement a survey + table structure for the future.
            if survey == 'DELS':
                cross_match_name = 'ls_dr9_tractor'

            # Set up cross-match name
            filename = '{}/{}_{}.parquet'.format(self.temp_dir,
                                                 cross_match_name,
                                                 idx)

            if os.path.exists(filename):
                msgs.info('Partition {} cross-match exists. Continuing to '
                          'next partition.'.format(idx))
            else:

                if survey == 'DELS':

                    # TODO: Check if match exists

                    cross_match = self.datalab_cross_match(
                        'ls_dr9.tractor',
                        columns,
                        match_distance=match_distance)

                # Save cross-matched chunk to temporary folder
                cross_match_df = cross_match.df.compute()
                cross_match_df.to_parquet(filename)
                msgs.info('Downloaded cross match {} to temporary '
                          'folder'.format(idx))

        # Log out of datalab
        if survey in ['DELS']:
            logout_status = ac.logout()
            msgs.info('Log out of NOIRLAB Astro Data Lab'
                      ' - Status: {}'.format(logout_status))

        # Merge downloaded tables with source table
        match = dd.read_parquet(self.temp_dir)

        # Merge dataframes
        msgs.info('Creating cross-matched dataframe.')
        merge = self.df.merge(match,
                     left_on=self.id_colname,
                     right_on='source_id',
                     how='left',
                     suffixes=('_source', '_match'))

        # Save merged dataframe
        msgs.info('Saving cross-matched dataframe to {}'.format(
            cross_match_name))

        merge.to_parquet('{}'.format(cross_match_name))

        # Remove the temporary folder (default)
        if self.clear_temp_dir:
            msgs.info('Removing temporary folder ({})'.format(self.temp_dir))
            shutil.rmtree(self.temp_dir)

    def merge_catalog_on_column(self, input_catalog, left_on, right_on):

        merge = self.df.merge(input_catalog,
                              left_on=left_on,
                              right_on=right_on,
                              how='left',
                              suffixes=('_source', '_match'))

        # Save merged dataframe
        msgs.info('Saving cross-matched dataframe to {}'.format(
            input_catalog.name))

        merge.to_parquet('{}'.format(input_catalog.name))


    def datalab_cross_match(self, datalab_table, columns, match_distance):
        """Cross-match catalog to online catalog from the NOIRLAB Astro data
        lab.

        Downloaded catalog data is saved in the temporary directory. Upon
        completion of the online cross-match, the downloaded catalog data is
        merged to the input catalog and saved.

        :param datalab_table: Datalab table identifier
        :type datalab_table: string
        :param columns: List of column names to match
        :type columns: list
        :param match_distance: Match distance in arcseconds
        :type match_distance: float
        :return: None
        """

        # Convert match_distance to degrees
        match_distance = match_distance/3600.

        # Loading default column values
        # TODO expand this to more catalogs and take care of columns = None
        if datalab_table == 'ls_dr9.tractor' and columns == 'default':
            columns = whcd.ls_dr9_default_columns
            match_name = '{}_x_ls_dr9_tractor'.format(self.name)
        else:
            msgs.warn('Datalab Table not implemented yet')

        # Upload table to Astro Data Lab
        msgs.info('Uploading dataframe to Astro Data Lab')
        msgs.info('Upload status: '
                  + qc.mydb_import('wild_upload', self.chunk, drop=True))

        upload_table = 'mydb://wild_upload'
        survey = 'ls_dr9'
        datalab_table = 'tractor'

        # Build SQL query
        if columns is not None:
            sql_query = '''SELECT s.{} as source_id,
                                s.{} as source_ra,
                                s.{} as source_dec, '''.format(
                self.id_colname,
                self.ra_colname,
                self.dec_colname)
            sql_query += '{}'.format(columns)
        else:
            sql_query = 'SELECT * '

        sql_query += \
            '(q3c_dist(s.ra, s.dec, match.ra, match.dec)*3600) as dist_arcsec '

        sql_query += 'FROM {} AS s '.format(upload_table)

        sql_query += 'LEFT JOIN LATERAL ('

        sql_query += 'SELECT g.* FROM {}.{} AS g '.format(survey, datalab_table)

        sql_query += 'WHERE q3c_join(s.ra, s.dec, g.ra, g.dec, {}) '.format(
            match_distance)

        sql_query += 'ORDER BY q3c_dist(s.ra,s.dec,g.ra,g.dec) ASC LIMIT 1) '

        sql_query += 'AS match on true'

        if self.verbose > 1:
            msgs.info('SQL query: \n {}'.format(sql_query))

        response = qc.query(sql=sql_query, fmt='csv')

        result_df = convert(response)

        cross_match = Catalog(match_name,
                              id_column_name=self.id_colname,
                              ra_column_name=self.ra_colname,
                              dec_column_name=self.dec_colname,
                              table_data=result_df)

        return cross_match

    def get_offset_stars_datalab(self, match_distance, datalab_dict,
                                 n=3, where=None, minimum_distance=3):
        """Get offset stars for all targets in the catalog from the NOIRLAB
        Astro data lab.

        Although this routine works with large catalogs, it is slow in
        the current implementation.

        Example of a datalab_dict

        dict = {'table': 'ls_dr9.tractor',
                'ra': 'ra',
                'dec': 'dec',
                'mag': 'mag_z',
                'mag_name': 'lsdr9_z'}

        :param match_distance: Maximum search radius in arcseconds
        :type match_distance: float
        :param datalab_dict: Dictionary with keywords to identify offset
        star information.
        :type: dict
        :param n: Number of offset stars to retrieve. (Maximum: n=5)
        type n: int
        :param where: string
            A string written in ADQL syntax to apply quality criteria on
            potential offset stars around the target.
        :param minimum_distance: Minimum distance to the target in arcsec
        :type minimum_distance: float
        :return: None
        """

        msgs.info('Retrieving offset stars for catalog from NOIRLAB Astro '
                  'Datalab.')
        # Restrict the number of offset stars to be returned to n=5
        if n > 5:
            n = 5
        elif n < 0:
            n = 1

        # Serialized offset star query over all catalog partitions
        for idx, partition in enumerate(self.df.partitions):
            msgs.info('Working on partition {}'.format(idx+1))

            df = partition.compute()
            offset_df = pd.DataFrame()

            # Loop over each individual target in the catalog
            # Warning: This is slow for very large catalogs.
            # ToDo: Implement a faster query at some point.
            for jdx in df.index:
                target_name = df.loc[jdx, self.id_colname]
                target_ra = df.loc[jdx, self.ra_colname]
                target_dec = df.loc[jdx, self.dec_colname]

                temp_df = whcq.get_datalab_offset(
                    target_name, target_ra, target_dec, match_distance,
                    datalab_dict, columns=None,
                    where=where, n=n, minimum_distance=minimum_distance,
                    verbosity=self.verbose)

                offset_df = pd.concat([offset_df, temp_df], ignore_index=True)

                offset_df.to_csv('temp_offset_df.csv', index=False)

            # Remove temporary backup file
            os.remove('temp_offset_df.csv')

            if self.df.npartitions > 1:
                filename = '{}_{}_OFFSETS_part_{}.csv'.format(
                    self.name, datalab_dict['table'].split('.')[0], idx)
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)
            else:
                filename = '{}_{}_OFFSETS.csv'.format(
                    self.name, datalab_dict['table'].split('.')[0])
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)

    def get_offset_stars_astroquery(self, match_distance, catalog='tmass',
                                    n=3, quality_query=None,
                                    minimum_distance=3):
        """Get offset stars for all targets in the catalog using astroquery.

        Although this routine works with large catalogs, it will be slow in
        the current implementation.

        :param match_distance: float
            Maximum search radius in arcseconds
        :param catalog: string
            Catalog (and data release) to retrieve the offset star data from.
            See astroquery_dict for implemented catalogs.
        :param n: int
            Number of offset stars to retrieve. (Maximum: n=5)
        :param quality_query: string
            A string written in pandas query syntax to apply quality criteria
            on potential offset stars around the target.
        :param minimum_distance: Minimum distance to the target in arcsec
        :type minimum_distance: float
        :return: None
        """
        msgs.info('Retrieving offset stars for catalog using astroquery.')

        # Restrict the number of offset stars to be returned to n=5
        if n > 5:
            n = 5
        elif n < 0:
            n = 1

        # Serialized offset star query over all catalog partitions
        for idx, partition in enumerate(self.df.partitions):
            msgs.info('Working on partition {}'.format(idx + 1))

            df = partition.compute()
            offset_df = pd.DataFrame()

            # Loop over each individual target in the catalog
            # Warning: This is slow for very large catalogs.
            # ToDo: Implement a faster query at some point.
            for jdx in df.index:
                target_name = df.loc[jdx, self.id_colname]
                target_ra = df.loc[jdx, self.ra_colname]
                target_dec = df.loc[jdx, self.dec_colname]

                temp_df = whcq.get_astroquery_offset(
                    target_name, target_ra, target_dec,
                    match_distance, catalog,
                    quality_query=quality_query, n=n,
                    minimum_distance=minimum_distance,
                    verbosity=self.verbose)

                offset_df = pd.concat([offset_df, temp_df], ignore_index=True)

                offset_df.to_csv('temp_offset_df.csv', index=False)

            # Remove temporary backup file
            os.remove('temp_offset_df.csv')

            if self.df.npartitions > 1:
                filename = '{}_{}_OFFSETS_part_{}.csv'.format(
                    self.name, catalog, idx)
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)
            else:
                filename = '{}_{}_OFFSETS.csv'.format(
                    self.name, catalog)
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)

    def get_offset_stars_ps1(self, radius, data_release='dr2',
                             catalog='mean', quality_query=None, n=3):
        """Get offset stars for all targets in the catalog for PanSTARRS
        using the MAST website.

        Currently, this runs slowly as it queries the PanSTARRS 1 archive for
        each object individually. But it runs!

        It will always retrieve the z-band magnitude for the offset star. This
        is hardcoded in get_ps1_offset_star(). Depending on the catalog it
        will be the mean of stack magnitude.

        :param radius: Maximum search radius in arcseconds
        :type radius: float
        :param data_release: The specific PanSTARRS data release (default: dr2)
        :type data_release: string
        :param catalog: Catalog to retrieve the offset star data from.
            (e.g. 'mean', 'stack')
        :type catalog: string
        :param quality_query: A string written in pandas query syntax to apply
            quality criteria on potential offset stars around the target.
        :param n: int
            Number of offset stars to retrieve. (Maximum: n=5)
        :return:
        """

        msgs.info('Retrieving offset stars for catalog from PanSTARRS1.')

        # Restrict the number of offset stars to be returned to n=5
        if n > 5:
            n = 5
        elif n < 0:
            n = 1

        # Serialized offset star query over all catalog partitions
        for idx, partition in enumerate(self.df.partitions):
            msgs.info('Working on partition {}'.format(idx + 1))

            df = partition.compute()
            offset_df = pd.DataFrame()

            # Loop over each individual target in the catalog
            # Warning: This is slow for very large catalogs.
            # ToDo: Implement a faster query at some point.
            for jdx in df.index:
                target_name = df.loc[jdx, self.id_colname]
                target_ra = df.loc[jdx, self.ra_colname]
                target_dec = df.loc[jdx, self.dec_colname]

                temp_df = whcq.get_ps1_offset_star(
                    target_name, target_ra, target_dec,
                    radius=radius, catalog=catalog,
                    data_release=data_release,
                    quality_query=quality_query, n=n,
                    verbosity=self.verbose)

            # Remove temporary backup file
            os.remove('temp_offset_df.csv')

            if self.df.npartitions > 1:
                filename = '{}_ps1_OFFSETS_part_{}.csv'.format(
                    self.name, idx)
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)
            else:
                filename = '{}_ps1_OFFSETS.csv'.format(
                    self.name)
                msgs.info('Saving offset stars to {}'.format(filename))
                offset_df.to_csv(filename)

    def get_survey_images(self, image_folder_path, survey_dicts, n_jobs=1):
        """ Retrieve survey images from an online imaging survey.

        :param image_folder_path: Storage path for the downloaded images.
        :type image_folder_path: string
        :param survey_dicts: Survey dictionaries
        :type survey_dicts: dict
        :param n_jobs: Number of parallel jobs
        :type n_jobs: int
        :return: None
        """

        for survey_dict in survey_dicts:
            survey = retrieve_survey(survey_dict['survey'],
                                     survey_dict['bands'],
                                     survey_dict['fov'])

            for partition in self.df.partitions:
                ra = partition.compute()[self.ra_colname]
                dec = partition.compute()[self.dec_colname]
                survey.download_images(ra, dec, image_folder_path, n_jobs)

    def get_forced_photometry(self, image_folder_path, survey_dict, n_jobs=1):

        # Dummy function for development.

        pass



