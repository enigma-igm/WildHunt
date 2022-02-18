#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table

from urllib.request import urlopen  # python3
from urllib.error import HTTPError
from http.client import IncompleteRead

from wildhunt.surveys import imagingsurvey
from wildhunt import utils


class Panstarrs(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

        super(Panstarrs, self).__init__(bands, fov, 'ps1', verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """
        # Check if download directory exists. If not, create it
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)

        self.obj_names = utils.coord_to_name(ra, dec, epoch="J")

        for band in self.bands:

            for idx, obj_name in enumerate(self.obj_names):

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                           band + "_fov" + '{:d}'.format(self.fov)

                # Get url
                url = self.get_ps1_image_cutout_url(ra[idx], dec[idx],
                                                    fov=self.fov,
                                                    bands=band)

                print(url)

                self.download_image_from_url(url[0], image_name,
                                             image_folder_path)

    def download_image_from_url(self, url, image_name, image_folder_path):

        # Try except clause for downloading the image
        try:
            datafile = urlopen(url)

            check_ok = datafile.msg == 'OK'

            if check_ok:

                file = datafile.read()

                output = open(image_folder_path + '/' + image_name + '.fits', 'wb')
                output.write(file)
                output.close()
                if self.verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_folder_path))

        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            print(err)
            if self.verbosity > 0:
                print("Download of {} unsuccessful".format(image_name))

    def get_ps1_filenames(self, ra, dec, bands='g'):
        """

        :param ra:
        :param dec:
        :param bands:
        :return:
        """
        url_base = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'
        ps1_url = url_base + 'ra={}&dec={}&filters={}'.format(ra, dec, bands)

        table = Table.read(ps1_url, format='ascii')

        # Sort filters from red to blue
        flist = ["yzirg".find(x) for x in table['filter']]
        table = table[np.argsort(flist)]

        filenames = table['filename']

        if len(filenames) > 0:
            return filenames
        else:
            print("No PS1 image is available for this position.")
            return None

    def get_ps1_image_cutout_url(self, ra, dec, fov, bands='g', verbosity=0):
        """

        :param ra:
        :param dec:
        :param fov:
        :param bands:
        :param verbosity:
        :return:
        """

        # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
        size = fov * 4

        filenames = self.get_ps1_filenames(ra, dec, bands)

        if filenames is not None:

            url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                   "ra={ra}&dec={dec}&size={size}&format=fits").format(
                **locals())

            urlbase = url + "&red="
            url_list = []
            for filename in filenames:
                url_list.append(urlbase + filename)

            return url_list
        else:
            return None

# BULKD DOWNLOAD TO IMPLEMENT SEE BELOW!

# import numpy as np
# from astropy.table import Table
# import requests
# import time
# from io import StringIO
#
# ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
# fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
#
#
# def getimages(tra, tdec, size=240, filters="grizy", format="fits",
#               imagetypes="stack"):
#     """Query ps1filenames.py service for multiple positions to get a list of images
#     This adds a url column to the table to retrieve the cutout.
#
#     tra, tdec = list of positions in degrees
#     size = image size in pixels (0.25 arcsec/pixel)
#     filters = string with filters to include
#     format = data format (options are "fits", "jpg", or "png")
#     imagetypes = list of any of the acceptable image types.  Default is stack;
#         other common choices include warp (single-epoch images), stack.wt (weight image),
#         stack.mask, stack.exp (exposure time), stack.num (number of exposures),
#         warp.wt, and warp.mask.  This parameter can be a list of strings or a
#         comma-separated string.
#
#     Returns an astropy table with the results
#     """
#
#     if format not in ("jpg", "png", "fits"):
#         raise ValueError("format must be one of jpg, png, fits")
#     # if imagetypes is a list, convert to a comma-separated string
#     if not isinstance(imagetypes, str):
#         imagetypes = ",".join(imagetypes)
#     # put the positions in an in-memory file object
#     cbuf = StringIO()
#     cbuf.write(
#         '\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra, tdec)]))
#     cbuf.seek(0)
#     # use requests.post to pass in positions as a file
#     r = requests.post(ps1filename, data=dict(filters=filters, type=imagetypes),
#                       files=dict(file=cbuf))
#     r.raise_for_status()
#     tab = Table.read(r.text, format="ascii")
#
#     urlbase = "{}?size={}&format={}".format(fitscut, size, format)
#     tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase, ra, dec, filename)
#                   for (filename, ra, dec) in
#                   zip(tab["filename"], tab["ra"], tab["dec"])]
#     return tab
#
#
# if __name__ == "__main__":
#     t0 = time.time()
#
#     # create a test set of image positions
#     tdec = np.append(np.arange(31) * 3.95 - 29.1, 88.0)
#     tra = np.append(np.arange(31) * 12., 0.0)
#
#     # get the PS1 info for those positions
#     table = getimages(tra, tdec, filters="ri")
#     print("{:.1f} s: got list of {} images for {} positions".format(
#         time.time() - t0, len(table), len(tra)))
#
#     # extract cutout for each position/filter combination
#     for row in table:
#         ra = row['ra']
#         dec = row['dec']
#         projcell = row['projcell']
#         subcell = row['subcell']
#         filter = row['filter']
#
#         # create a name for the image -- could also include the projection cell or other info
#         fname = "t{:08.4f}{:+07.4f}.{}.fits".format(ra, dec, filter)
#
#         url = row["url"]
#         print("%11.6f %10.6f skycell.%4.4d.%3.3d %s" % (
#         ra, dec, projcell, subcell, fname))
#         r = requests.get(url)
#         open(fname, "wb").write(r.content)
#     print("{:.1f} s: retrieved {} FITS files for {} positions".format(
#         time.time() - t0, len(table), len(tra)))