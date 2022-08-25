#!/usr/bin/env python

import os

import getpass
import shutil
import numpy as np
import pandas as pd

import glob
import dask
import dask.dataframe as dd

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import ascii, fits


# from dl import authClient as ac, queryClient as qc, storeClient as sc
# from dl.helpers.utils import convert



from IPython import embed

from wildhunt import catalog_defaults
from wildhunt import pypmsgs
from wildhunt import utils

msgs = pypmsgs.Messages()

# Potentially instantiate a derived class, called SegmentedCatalog


class Catalog(object):
    """ Catalog class to handle small to large operations on photometric
    catalogs.

    Catalogs are operated on internally as dask dataframes to guarantee
    memory and computation efficient catalog operations.

    The standard file format is parquet. As a default catalogs are stored
    in folders with N parquet files according to the number of dask
    dataframe partitions.

    In the case of

    """

    def __init__(self, name, ra_column_name, dec_column_name,
                 id_column_name, datapath=None, table_data=None,
                 chunksize=10000, verbose=1,
                 clear_temp_dir=True, dtype=None):


        # Main attributes
        self.name = name
        self.datapath = datapath
        self.df = None
        self.verbose = verbose
        self.dtype = dtype

        # Column name attributes
        self.id_colname = id_column_name
        self.ra_colname = ra_column_name
        self.dec_colname = dec_column_name
        self.columns = [self.id_colname, self.ra_colname, self.dec_colname]

        # Attributes for file
        self.chunksize = chunksize
        self.chunk = None  # Dataframe of the current batch
        self.temp_dir = 'wild_temp'
        self.clear_temp_dir = clear_temp_dir

        # Attributes for file partitioning
        self.partition_limit = 1073741824 # 1GB
        self.partition_size = 104857600 # 100Mb

        # Instantiate dask dataframe
        if table_data is None and datapath is None:
            msgs.error('No dataframe or datapath specified.')
        elif table_data is None and datapath is not None:
            self.init_dask_dataframe()
        elif datapath is None and table_data is not None:
            self.df = dd.from_pandas(table_data, npartitions=1)

    # def _get_iterator(self, full=False):
    #
    #     file_suffix = self.datapath.split('.')[-1]
    #
    #     if file_suffix == 'csv':
    #         if not full:
    #             iterator = pd.read_csv(self.datapath, usecols=self.columns,
    #                                    iterator=True,
    #                                    chunksize=self.chunksize)
    #         else:
    #             iterator = pd.read_csv(self.datapath,
    #                                    iterator=True,
    #                                    chunksize=self.chunksize)
    #     elif file_suffix == 'hdf5':
    #         iterator = pd.read_hdf(self.datapath,
    #                                iterator=True,
    #                                chunksize=self.chunksize)
    #
    #     return iterator

    def init_dask_dataframe(self):

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
                n_partitions = round(filesize / self.partition_size)
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
                msgs.warn('Converting a large fits file.')
                table = Table.read(self.datapath)
                df = table.to_pandas()
                # df.to_parquet('test', index=False)
                df = dd.from_pandas(df,
                                    npartitions=n_partitions)
                print(df)
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


    # def load_catalog(self, full=False):
    #
    #
    #     if self.datapath is None:
    #         raise ValueError('[ERROR] Filename not defined!')
    #
    #     file_suffix = self.datapath.split('.')[-1]
    #
    #     if file_suffix == 'csv':
    #         if not full:
    #             df = pd.read_csv(self.datapath, usecols=self.columns)
    #         else:
    #             df = pd.read_csv(self.datapath)
    #     elif file_suffix == 'hdf5':
    #         df = pd.read_hdf(self.datapath)
    #     elif file_suffix == 'parquet':
    #         if not full:
    #             df = pd.read_parquet(self.datapath, columns=self.columns)
    #         else:
    #             df = pd.read_parquet(self.datapath)
    #     elif file_suffix == 'fits':
    #         tbl = Table()
    #         tbl.read(self.datapath)
    #         df = tbl.to_pandas()
    #     else:
    #         raise ValueError('File suffix not supported.')
    #
    #     if (file_suffix != 'parquet' or file_suffix !='csv') and not full:
    #         self.df = df[self.columns].copy()
    #     else:
    #         self.df = df.copy()

    def save_catalog(self, filename: str = None,
                     file_format: str = 'parquet') -> None:
        """

        :param filename:
        :param format:
        """
        if filename is None:
            filename = self.name

        if file_format == 'parquet':
            self.df.to_parquet('{}'.format(filename))
        elif file_format == 'csv':
            self.df.to_csv('{}'.format(filename), index=False)
        elif file_format == 'hdf5':
            self.df.to_hdf('{}'.format(filename), format='table')
        else:
            msgs.error('Provided file format {} not supported'.format(
                file_format))

    def catalog_cross_match(self, match_catalog, match_distance,
                            columns='all',
                            merged_cat_name='merged', column_prefix=None):

        msgs.info('Cross-matching {} with {}'.format(self.name,
                                                     match_catalog.name))
        msgs.warn('The current implementation does not make optimal use '
                  'of dask dataframes.')
        msgs.warn('In consequence the full match dataframe is loaded in '
                  'memory.')

        for idx, partition in enumerate(self.df.partitions):

            msgs.info('Matching partition {} out of {}'.format(
                idx+1, self.df.npartitions))

            # Create source pandas dataframe from partition
            source = partition.compute()
            # Create match pandas dataframe from partition
            match_df = match_catalog.df.compute()

            # # Usual astropy procedure
            source_coord = SkyCoord(
                source[self.ra_colname].values * u.deg,
                source[self.dec_colname].values * u.deg)
            match_coord = SkyCoord(
                match_df[match_catalog.ra_colname].values * u.deg,
                match_df[match_catalog.dec_colname].values * u.deg)

            match_idx, separation, d3d = match_coordinates_sky(
                source_coord,
                match_coord)

            # match_idx, separation, d3d = source_coord.match_to_catalog_sky(
            #     match_coord)

            # Add match_index and match distance to df
            source['match_index'] = match_idx
            source['{}_distance'.format(column_prefix)] = separation.to(
                'arcsec').value

            # Generate columsn for merge
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

            filename = '{}/{}_{}'.format(self.temp_dir, merged_cat_name,
                                         idx)
            df_merged.to_parquet(filename)


    def cross_match(self, survey='DELS', columns='default',
                    match_distance=3):
        """

        :param survey:
        :param columns:
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
            msgs.info('Log in to NOIRLAB Astro Data Lab')
            token = ac.login(input('Enter user name (+ENTER): '),
                             getpass.getpass('Enter password (+ENTER): '))
            msgs.info('Astro Data Lab USER: {}'.format(ac.whoAmI()))
            msgs.info('Astro Data Lab TABLES: {}'.format( qc.mydb_list()))


        # Chunk-wise query operation
        for idx, partition in enumerate(self.df.partitions):
            msgs.info('Beginning crossmatch on partition {}'.format(idx))
            self.chunk = partition.compute()[self.columns]

            if survey == 'DELS':

                # TODO: Check if match exists

                cross_match = self.datalab_cross_match(
                    'ls_dr9.tractor',
                    columns,
                    match_distance=match_distance)

            # Save cross-matched chunk to temporary folder
            filename = '{}/{}_{}'.format(self.temp_dir, cross_match.name,
                                         idx)
            cross_match.save_catalog(filename)
            msgs.info('Downloaded cross match {} to temporary '
                      'folder'.format(idx))

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
            cross_match.name))
        merge.to_parquet('{}'.format(cross_match.name))

        if self.clear_temp_dir:
            msgs.info('Removing temporary folder ({})'.format(self.temp_dir))
            shutil.rmtree(self.temp_dir)

    def datalab_cross_match(self, table, columns, match_distance):

        # Convert match_distance to degrees
        match_distance = match_distance/3600.

        # Loading default column values
        if table == 'ls_dr9.tractor' and columns == 'default':
            columns = catalog_defaults.ls_dr9_default_columns
            match_name = '{}_x_ls_dr9_tractor'.format(self.name)
        else:
            msgs.warn('Datalab Table not implemented yet')

        # Upload table to Astro Data Lab
        msgs.info('Uploading dataframe to Astro Data Lab')
        msgs.info('Upload status: '
                  + qc.mydb_import('wild_upload', self.chunk, drop=True))

        upload_table = 'mydb://wild_upload'
        survey = 'ls_dr9'
        table = 'tractor'

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

        sql_query += 'SELECT g.* FROM {}.{} AS g '.format(survey, table)

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