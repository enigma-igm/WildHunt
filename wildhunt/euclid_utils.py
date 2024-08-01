#!/usr/bin/env python
import base64
import getpass
import os
from http.cookiejar import MozillaCookieJar
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy import units
from astropy.coordinates import SkyCoord

from wildhunt import pypmsgs

msgs = pypmsgs.Messages()

# random notes from the archive (Updated 20240723)
# sedm.aux_stacked -> auxiliary data (PSF and other data model)
# sedm.calibrated_detectors -> information about the detectors (including zeropoints)
# sedm.calibrated_frame -> contains information about calibrated images (including paths)
#   This seem to also include the MER catalogue in the future (F-006)
# sedm.column_values -> unclear
# sedm.combined_spectra and sedm.combined_spectra_product -> spectral products, unclear in what shape or form
# sedm.mosaic_product -> mosaic including filters from different surveys - unclear what their use or utility is
# sedm.observation_mode -> useful for the IDs in any observation mode column -> see CSV file
# sedm.observation_mosaic -> id match for the mosaic - if needed, this needs some digging for the id match
# sedm.observation_stack -> all the info about the stacked images
#   This seem to be what we actually want to use!

# ivoa_score table is up to date as of 20240723

# NOTE! It seems that one can only be logged in through the terminal OR the web page
# !! If both, everything breaks !!

# https://stackoverflow.com/questions/30405867/how-to-get-python-requests-to-trust-a-self-signed-ssl-certificate
# as far as I understand strongly recommended but not strictly needed
if os.environ.get("WILDHUNT_LOCALPATH") is None:
    LOCAL_PATH = Path.home()
else:
    LOCAL_PATH = Path(os.environ.get("WILDHUNT_LOCALPATH"))

CERT_KEY = LOCAL_PATH / "eas-esac-esa-int-chain.pem"
if not CERT_KEY.exists():
    CERT_KEY = False
VERBOSE = 0

# =========================================================================== #


def b64e(s):
    return base64.b64encode(s.encode()).decode()


def b64d(s):
    return base64.b64decode(s).decode()


# =========================================================================== #


class User:
    # basic user to handle the login, nothing especially fancy
    # password is encoded in base64 just to prevent accidental prints
    def __init__(
        self,
        username=None,
        password=None,
        filepath=LOCAL_PATH / "sas_otf_user_data.cfg",
        encoded=True,
    ):
        # check whether we have a configuration file. If so, try to load the data from there
        if filepath.exists():
            with open(filepath, "r") as f:
                data = f.read().split()
                self.username = data[0]
                # it is was written, this was already encoded
                self.password = data[1] if encoded else b64e(data[1])
        else:
            self.username = (
                username if username is not None else input("Enter user name: ")
            )
            self.password = (
                b64e(password)
                if password is not None
                else b64e(getpass.getpass("Enter password: "))
            )
        self.login_data = {"username": self.username, "password": self.password}
        self.logged_in = False
        self.cookies = None

    # ======================================================================= #

    def load_user_data(self, filepath=LOCAL_PATH / "sas_otf_user_data.cfg"):
        try:
            with open(filepath, "r") as f:
                data = f.read().split()
                self.username = data[0]
                self.password = b64e(data[1])
        except FileNotFoundError:
            msgs.info(f"{filepath} not found!")

    # ======================================================================= #

    def set_user_data(self, force=False):
        if self.username is None or force:
            self.username = input("Enter user name: ")
        else:
            msgs.info("User already set, pass `force` to reset it.")

        if self.password is None or force:
            self.password = b64e(getpass.getpass("Enter password: "))
        else:
            msgs.info("Password already set, pass `force` to reset it.")

        self.login_data = {"username": self.username, "password": self.password}

    # ======================================================================= #

    def get_user_data(self):
        return {
            "username": self.login_data["username"],
            "password": b64d(self.login_data["password"]),
        }

    # ======================================================================= #

    def store_user_data(self, path=LOCAL_PATH, overwrite=False):
        outfile = path / "sas_otf_user_data.cfg"
        if outfile.exists() and not overwrite:
            user_overwrite = input("[Info] Table exists, overwrite? [y]/n ").lower()
            if user_overwrite in ["y", "\n"]:
                overwrite = True

        if not outfile.exists() or overwrite:
            with open(outfile, "w") as f:
                f.write(f"{self.username} {self.password}")

    # ======================================================================= #

    def sasotf_login(self, cert_key=CERT_KEY):
        if self.login_data is None:
            self.set_user_data()

        cookies = MozillaCookieJar()
        with requests.Session() as session:
            session.cookies = cookies

            session.post(
                "https://easotf.esac.esa.int/tap-server/login",
                data=self.get_user_data(),
                verify=cert_key,
            )

            session.post(
                "https://easotf.esac.esa.int/sas-cutout/login",
                data=self.get_user_data(),
                verify=cert_key,
            )
            # do I really need to save the cookies?
            # cookies.save("cookies.txt", ignore_discard=True, ignore_expires=True)

        self.logged_in = True
        self.cookies = cookies
        msgs.info("Log in to the Euclid OTF archive successful!")

    # ======================================================================= #

    def __str__(self):
        out = f"User {self.username}\n"
        if self.logged_in:
            out += "Logged in. If experiencing problems, reload the user data and/or try logging in again."
        else:
            out += "Not logged in."

        return out

    # ======================================================================= #

    def __repr__(self):
        # in this case don't really care about the ectra info - I just need to know the current user and if I am (possibly)
        #   logged in
        out = f"euclid_download.User()\nCurrent user: {self.username}\n"
        if self.logged_in:
            out += "User logged in. If experiencing problems, reload the user data and/or try logging in again."
        else:
            out += "User not logged in."

        return out


# =========================================================================== #
# =========================== Catalogue functions =========================== #
# =========================================================================== #


# For the time being, ONLY for cutout purposes, the best approach seems to be using ivoa_obscore with the calibrated images
# stacked images are available, but it appears only a subset of those are available from the archive (Deep fields? Unsure...)
def select_query(name="stack"):
    assert name in [
        "ivoa_obscore",
        "stack",
        "calib",
    ], "[Error] Valid options are `'ivoa_obscore', 'stack' and 'calib'`"

    if name == "ivoa_obscore":
        return """SELECT s_ra, s_dec, t_exptime, obs_id, obs_collection, cutout_access_url,
               dataproduct_subtype, dataproduct_type, filter, instrument_name
               FROM ivoa.obscore WHERE t_exptime > 0"""
    elif name == "stack":
        return """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, observation_stack_oid, product_type,
               release_name FROM sedm.observation_stack"""
    elif name == "calib":
        return """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, calibrated_frame_oid, product_type,
               release_name FROM sedm.calibrated_frame"""


# =========================================================================== #


def download_table(query_table, user, savepath, sync=True, verbose=VERBOSE):
    if not user.logged_in:
        msgs.info("User not logged in, trying log in.")
        user.sasotf_login()

    # minimal useful information (I think)
    query = select_query(query_table)

    if sync:
        response = requests.get(
            "https://easotf.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
            + query.replace(" ", "+"),
            verify=CERT_KEY,
            cookies=user.cookies,
        )

        if verbose > 0:
            msgs.info(f"Response code: {response.status_code}")

        with open(savepath, "w") as f:
            f.write(response.content.decode("utf-8"))

    # Async (already required for ivoa_obscore)
    # FIXME! How do I get the correct job if I do N queries?
    # For now just use a local table that we update every now and then
    # Current version is above
    if False:
        with requests.Session() as session:
            cookies = MozillaCookieJar()
            session.cookies = cookies

            session.post(
                "https://easotf.esac.esa.int/tap-server/tap/async",
                data={
                    "data": "PHASE=run&LANG=ADQL&REQUEST=doQuery&QUERY="
                    + query.replace(" ", "+")
                },
                verify=CERT_KEY,
            )

            print(
                requests.get(
                    "https://easotf.esac.esa.int/tap-server/tap/async"
                ).content.decode("utf-8")
            )

    return pd.read_csv(savepath)


# =========================================================================== #


def prepare_catalogue(tbl_in, inplace=False, force=False):
    if inplace:
        tbl = tbl_in
    else:
        tbl = tbl_in.copy()

    if "cutout_access_url" not in tbl.columns or force:
        msgs.info("Added `cutout_access_url` column.")
        # names are the same for both stack and calib, which is what we'll want to use most of the time
        tbl["cutout_access_url"] = [
            build_access_url(_p.strip(), _f.strip(), _o)
            for (_p, _f, _o) in zip(
                tbl["file_path"], tbl["file_name"], tbl["observation_id"]
            )
        ]

    # needed for ivoa_score
    if "s_ra" in tbl.columns:
        msgs.info("Renaming columns s_ra, s_dec, dataproduct_subtype and filter.")
        tbl.rename(
            {
                "s_ra": "ra",
                "s_dec": "dec",
                "dataproduct_subtype": "product_type",
                "filter": "filter_",
            },
            axis=1,
            inplace=True,
        )

    # needed for stack and calib
    if "filter_name" in tbl.columns:
        msgs.info("Renaming column filter_name.")
        tbl.rename(
            {"filter_name": "filter_"},
            axis=1,
            inplace=True,
        )

    return tbl


# =========================================================================== #


def load_catalogue(
    fname="", tbl_in=None, query_table="stack", user=None, overwrite=False
):
    if user is None:
        user = User()

    # either read from file (and optionally query and download)
    # or directly pass a table
    if fname != "":
        if Path(fname).exists() and not overwrite:
            msgs.info("Table exists, loading it.")
            tbl = pd.read_csv(fname)
        else:
            msgs.info(f"Querying archive and saving results to {fname}.")
            tbl = download_table(query_table, user, Path(fname))
    elif tbl_in is not None:
        tbl = tbl_in.copy()
    else:
        msgs.error(
            "Either provide a path to load a catalogue from, or the table itself."
        )

    return tbl, user


# =========================================================================== #


def build_access_url(path, filename, obsid):
    # this builds the first part of the url
    # get collection from path itself
    collection = path.split("/")[-2]
    base = "https://easotf.esac.esa.int/sas-cutout/cutout"
    params = f"filepath={path}/{filename}&collection={collection}&obsid={obsid}"
    return f"{base}?{params}"


# =========================================================================== #


def build_url(access_url, ra, dec, side, search_type="CIRCLE"):
    # this adds the search parameter
    search = f"POS={search_type},{ra},{dec},{side}"
    return f"{access_url.strip()}&{search.strip()}"


# =========================================================================== #


@units.quantity_input()
def get_closest_image_url(
    # ra: units.deg, dec: units.deg, side : units.arcsec, cat, ra_cat="ra", dec_cat="dec"
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
):
    # this takes the closes images to the target
    # TODO: Are these unique?
    target_coord = SkyCoord(ra, dec, frame="icrs")
    cat_coord = SkyCoord(
        cat[ra_cat].to_numpy() * ra.unit,
        cat[dec_cat].to_numpy() * dec.unit,
        frame="icrs",
    )
    dist = target_coord.separation(cat_coord).to(units.arcsec)
    if len(dist) == 0:
        return [], -1.0

    # probably too much
    # TODO: Is there a better way to determine the distance?
    elif dist.min() > 1.0 * units.deg:
        msgs.warn(
            f"Images in band {band} are all farther than 1 deg "
            f"for the object at ra, dec: {ra.value:.4f}, {dec.value:.4f}."
        )
        return [], dist.min()

    # now this is not ideal but at the moment I really don't
    #  see other simple options
    # images are selected based on the closest match.
    #  if needed, the code can download an arbitrary number of images, but
    #  names need to be adjusted accordingly, and this is NOT handled at
    #  the moment.
    inds = np.where(dist < np.unique(np.sort(dist))[1])[0]

    return cat["cutout_access_url"][inds], dist[inds]


# =========================================================================== #


@units.quantity_input()
def get_download_urls(
    ra: units.deg, dec: units.deg, side: units.arcsec, cat, band, search_type="CIRCLE"
):
    side = side.to(units.deg).value
    image_urls, _ = get_closest_image_url(ra, dec, cat, band)
    return [
        build_url(url, ra.value, dec.value, side, search_type) for url in image_urls
    ]


# =========================================================================== #


@units.quantity_input()
def get_download_df(
    ra_arr: units.deg, dec_arr: units.deg, side: units.arcsec, cat, requested_band
):
    # transition layer to wildhunt
    filtered_cat = cat.query(f"filter_ == '{requested_band}'").reset_index()

    urls_, filter_, ras, decs = [], [], [], []
    for _ra, _dec in zip(ra_arr, dec_arr):
        urls = get_download_urls(_ra, _dec, side, filtered_cat, requested_band)
        bands = [requested_band] * len(urls)

        # build columns for the dataframe
        urls_.append(urls)
        filter_.append(bands)
        ras.append([_ra.value] * len(urls))
        decs.append([_dec.value] * len(urls))

    return pd.DataFrame(
        data={
            "ra": np.hstack(ras),
            "dec": np.hstack(decs),
            "url": np.hstack(urls_),
            "filter": np.hstack(filter_),
        }
    )


# =========================================================================== #


def download_cutouts(obj_ra, obj_dec, img_urls, folder, user=None, verbose=VERBOSE):
    if user is None:
        user = User()
        user.sasotf_login()

    for img_url in img_urls:
        # will this be sufficient?
        if "NIR" in img_url:
            band = img_url.split("IMAGE_")[1][0]
        else:
            band = "VIS"

        if VERBOSE > 0:
            msgs.info(f"Image URL: {img_url}")

        response = requests.get(img_url, verify=CERT_KEY, cookies=user.cookies)
        if verbose > 0:
            msgs.info(f"Image download response code: {response.status_code}")

        # if there is actual content in the image I downloaded
        if len(response.content) == 0:
            msgs.info("Empty fits content, skipping.")

        if response.status_code == 200:
            with open(
                folder / f"{obj_ra:.5f}_{obj_dec:.5f}_{band}.fits",
                "wb",
            ) as f:
                f.write(response.content)
