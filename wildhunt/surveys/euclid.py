#!/usr/bin/env python

import multiprocessing as mp
import os
from http.client import IncompleteRead
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import requests
from astropy import units

from wildhunt import euclid_utils as eu
from wildhunt import pypmsgs, utils
from wildhunt.surveys import imagingsurvey

msgs = pypmsgs.Messages()
local_path = os.environ.get("WILDHUNT_LOCALPATH")


class Euclid(imagingsurvey.ImagingSurvey):
    """Euclid class deriving from the ImagingSurvey class to handle
    image downloads and aperture photometry for the Euclid survey.

    """

    def __init__(self, bands, fov, name="PS1", verbosity=1):
        """Initialize the Euclid class.

        :param bands: List of survey filter bands (VIS/Y/J/H).
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        :return: None
        """
        assert set(bands) <= set(
            [
                "VIS",
                "Y",
                "J",
                "H",
            ]
        ), "Valid bands for Euclid are VIS, Y, J, H"
        assert isinstance(bands, list), "`bands` argument should be a list"

        self.batch_size = 10000

        self.ra = None
        self.dec = None
        self.nbatch = 1

        user = eu.User()
        user.sasotf_login()

        self.user = user
        super(Euclid, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """Download images from the online image server for Euclid.

        :param ra: Right ascension of the sources in decimal degrees.
        :type ra: numpy.ndarray
        :param dec: Declination of the sources in decimal degrees.
        :type dec: numpy.ndarray
        :param image_folder_path: Path to the folder where the images are
         stored.
        :type image_folder_path: str
        :param n_jobs: Number of parallel jobs to use for downloading.
        :type n_jobs: int
        :return: None
        """
        if self.n_jobs > 1:
            msgs.warn(
                "Multiprocessing download is currently not working for Euclid."
                "Setting `self.n_jobs` to 1."
            )
            self.n_jobs = 1

        self.survey_setup(ra, dec, image_folder_path, epoch="J", n_jobs=n_jobs)

        if self.source_table.shape[0] > 0:
            self.batch_setup()

            for i in range(self.nbatch):
                self.retrieve_image_url_list(product_type="stacked", batch_number=i)

                self.check_for_existing_images_before_download()

                if self.n_jobs > 1:
                    # TODO: Do I need to give you the user as well?
                    self.mp_download_image_from_url()

                else:
                    for idx in self.download_table.index:
                        image_name = self.download_table.loc[idx, "image_name"]
                        url = self.download_table.loc[idx, "url"]
                        # TODO: Do I need to give you the user as well?
                        self.download_image_from_url(url, image_name)

            for i in range(self.nbatch):
                os.remove(str(i) + "_Euclid_download_urls.csv")

        else:
            msgs.info("All images already exist.")
            msgs.info("Download canceled.")

    def batch_setup(
        self,
    ):
        self.ra = self.source_table.loc[:, "ra"].values
        self.dec = self.source_table.loc[:, "dec"].values

        # Compute the number of batches to retrieve the urls
        if np.size(self.ra) > self.batch_size:
            self.nbatch = int(np.ceil(np.size(self.ra) / self.batch_size))

    def retrieve_image_url_list(
        self, batch_number=0, catalogue="ivoa_obscore", product_type="stacked"
    ):
        """Retrieve the list of image URLs from the online image server.

        :param batch_number: Number of the batch to retrieve the urls for.
        :type batch_number: int
        :param Catalogue: Catalogue from to download images from. Defaults to ivoa_obscore ("ivoa_obscore").
        :param product_type: Euclid product type [stacked, calib]
        :return: None
        """
        # ============================================ #
        # !!! THIS IS VERY MUCH A TEMPORARY SOLUTION !!!
        # ============================================ #

        # Catalogue for stacked and calib frames can (at this stage) be downloaded through sync queries
        #  Async quesries are an issue which we will worry about later on.
        # ivoa_obscore already takes too long as far as I can tell, so for the time being I am downloading
        #  it separately, caching it and calling it a day.

        # Euclid wants the image side directly. Arcseconds!
        img_size = self.fov
        bands = self.bands

        # There is also a DpdVisCalibratedQuadFrame that is unknown to me at the moment
        # TODO: Figure this out
        product_type_dict = {
            "calib": ["DpdVisCalibratedFrame", "DpdNirCalibratedFrame"],
            "stacked": ["DpdVisStackedFrame", "DpdNirStackedFrame"],
        }

        # Retrieve bulk file table
        # url_ps1filename = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py'

        ra_batch = self.ra[
            batch_number * self.batch_size : batch_number * self.batch_size
            + self.batch_size
        ]
        dec_batch = self.dec[
            batch_number * self.batch_size : batch_number * self.batch_size
            + self.batch_size
        ]

        # use eu.get_download_df to get the equivalent of PanSTARRS.py here
        cat = eu.load_catalogue(
            fname=local_path + "/ivoa_frames.csv",
            query_table=catalogue,
            user=self.user,
            overwrite=False,
        )[0]

        # select only the products that you want
        eu.prepare_catalogue(cat, inplace=True)
        cat_vis_stack = cat.query(
            f"product_type == '{product_type_dict[product_type][0]}'"
        ).reset_index()
        cat_nir_stack = cat.query(
            f"product_type == '{product_type_dict[product_type][1]}'"
        ).reset_index()

        # Need to split this in two as otherwise there way to know
        #  whether the N closest images are all in VIS or NIR
        # In this way instead I am sure I am getting at least one
        #  of each, if present.
        df_vis, df_nir = None, None
        if "VIS" in bands:
            df_vis = eu.get_download_df(
                ra_batch * units.deg,
                dec_batch * units.deg,
                img_size * units.arcsec,
                cat_vis_stack,
                bands,
            )

        if any([b in ["Y", "J", "H"] for b in bands]):
            df_nir = eu.get_download_df(
                ra_batch * units.deg,
                dec_batch * units.deg,
                img_size * units.arcsec,
                cat_nir_stack,
                bands,
            )

        df_list = [df for df in [df_vis, df_nir] if df is not None]
        df = pd.concat(df_list, ignore_index=True)

        # sort by RA - needed because I am querying separately
        #  VIS and NIR
        df.sort_values("ra", axis=0, inplace=True)

        # Group table by filter
        groupby = df.groupby(by="filter", sort=False)

        for group_key in groupby.groups.keys():
            group_df = groupby.get_group(group_key)
            band = group_key

            for idx in group_df.index:
                obj_name = utils.coord_to_name(
                    group_df.loc[idx, "ra"], group_df.loc[idx, "dec"]
                )[0]

                # Create image name
                image_name = (
                    obj_name
                    + "_"
                    + self.name
                    + "_"
                    + band
                    + "_fov"
                    + "{:d}".format(self.fov)
                )

                new_entry = pd.DataFrame(
                    data={"image_name": image_name, "url": group_df.loc[idx, "url"]},
                    index=[0],
                )
                self.download_table = pd.concat(
                    [self.download_table, new_entry], ignore_index=True
                )

        self.download_table.to_csv(
            "{}_Euclid_download_urls.csv".format(str(batch_number))
        )

    def download_image_from_url(self, url, image_name):
        """Download the image with name image_name from the given url.

        :param url: URL of the image to download.
        :type url: str
        :param image_name: Name of the image to download.
        :type image_name: str
        :return:
        """
        # need to overwrite this due to how manual this process is
        # in essence, try to download all images and just keep those
        # that are not empty. Closest images to source get priority.
        # For the rest, same function as parent class
        try:
            r = requests.get(url, cookies=self.user.cookies)
            if len(r.content) == 0:
                return

            open(self.image_folder_path + "/" + image_name + ".fits", "wb").write(
                r.content
            )

            if self.verbosity > 0:
                msgs.info(
                    "Download of {} to {} completed".format(
                        image_name, self.image_folder_path
                    )
                )

        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            msgs.warn("Download error encountered: {}".format(err))
            if self.verbosity > 0:
                msgs.warn("Download of {} unsuccessful".format(image_name))

    def mp_download_image_from_url(self):
        """Execute image download in parallel.

        :return: None
        """
        raise NotImplementedError
        # FIXME: Complaints about something that cannot be pickled
        mp_args = list(
            zip(
                self.download_table.loc[:, "url"].values,
                self.download_table.loc[:, "image_name"].values,
            )
        )

        with mp.Pool(processes=self.n_jobs) as pool:
            pool.starmap(self.download_image_from_url, mp_args)

    def force_photometry_params(self, header, band, filepath=None):
        """Set parameters to calculate aperture photometry for the Pan-STARRS1
        survey imaging.

        :param header: Image header
        :type header: astropy.io.fits.header.Header
        :param band: The filter band of the image
        :type band: str
        :param filepath: File path to the image
        :type filepath: str

        :return: None
        """
        raise NotImplementedError
        # TODO: Exptime is in the main table, and it seems that zeropoints
        #  are in the fits file itself. Maybe find a way to use both information?
        #  Should ask JT how difficult this would be...
        zpt = {
            "g": 25.0,
            "r": 25.0,
            "i": 25.0,
            "z": 25.0,
            "y": 25.0,
        }

        self.exp = header["EXPTIME"]
        self.back = True
        self.zpt = zpt[band]
        self.ab_corr = 0.0
        self.nanomag_corr = np.power(10, 0.4 * (22.5 - self.zpt))
