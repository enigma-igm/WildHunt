#!/usr/bin/env python
import base64
import getpass
import os
import re
import time
from http.client import IncompleteRead
from http.cookiejar import MozillaCookieJar
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from wildhunt import image as whimg
from wildhunt import pypmsgs
from wildhunt import utils as whut

msgs = pypmsgs.Messages()

ChunkedEncodingError = requests.exceptions.ChunkedEncodingError
ProtocolError = requests.urllib3.exceptions.ProtocolError

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
# following discussion with Arv and the general Euclid Hack Days, this now queries directly from the correct table
#  and directly picks the right image by first sorting by distance

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


product_type_dict = {
    "calib": ["DpdVisCalibratedFrame", "DpdNirCalibratedFrame"],
    "stacked": ["DpdVisStackedFrame", "DpdNirStackedFrame"],
    "mosaic": ["DpdMerBksMosaic", "DpdMerBksMosaic"],  # same for VIS and NIR
}

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

            # possibly this is not needed?
            # TODO: Investigate
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

            session.post(
                "https://easotf.esac.esa.int/sas-dd/login",
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
# also stacked images are dithered, and that is a bit of a mess to deal with anyway
def select_query(name="stack"):
    name = name.lower()

    if name not in ["ivoa_obscore", "stack", "calib", "mosaic"]:
        raise ValueError(
            "[Error] Valid options are `'ivoa_obscore', 'stack', 'calib', 'mosaic'`"
        )

    if name == "ivoa_obscore":
        out = """SELECT s_ra, s_dec, t_exptime, obs_id, obs_collection, cutout_access_url,
               dataproduct_subtype, dataproduct_type, filter, instrument_name
               FROM ivoa.obscore WHERE t_exptime > 0"""
    elif name == "stack":
        out = """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, observation_stack_oid, product_type,
               release_name FROM sedm.observation_stack"""
    elif name == "calib":
        out = """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, calibrated_frame_oid, product_type,
               release_name FROM sedm.calibrated_frame"""
    elif name == "mosaic":
        out = """SELECT ra, dec, file_name, file_path, filter_name,
               instrument_name, tile_index, mosaic_product_oid, product_type,
               release_name FROM sedm.mosaic_product"""

    return re.sub(r"\n +", " ", out)


# =========================================================================== #


def get_phase(jobID, session):
    return requests.get(
        f"https://easotf.esac.esa.int/tap-server/tap/async/{jobID}/phase",
        cookies=session.cookies,
    ).content.decode()


# =========================================================================== #


def sync_query(query, user, savepath, cert_key=CERT_KEY, verbose=VERBOSE):
    response = requests.get(
        "https://easotf.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
        + query.replace(" ", "+"),
        verify=cert_key,
        cookies=user.cookies,
    )

    if verbose > 0:
        msgs.info(f"Response code: {response.status_code}")

    if response.status_code == 200:
        if verbose:
            msgs.info(f"Successfully retrieved table, writing to {savepath}.")

        with open(savepath, "w") as f:
            if verbose:
                msgs.info(f"Saving table to {savepath}")
            f.write(response.content.decode("utf-8"))
    else:
        msgs.error(
            f"Something went wrong. Query returned status code {response.status_code}"
        )
        raise ValueError("Failed query, please retry.")


# =========================================================================== #


def async_query(query, user, savepath, cert_key=CERT_KEY, verbose=VERBOSE):
    with requests.Session() as session:
        session.cookies = user.cookies

        post_sess = session.post(
            "https://easotf.esac.esa.int/tap-server/tap/async",
            data={
                "data": "PHASE=run&REQUEST=doQuery",
                "QUERY": query,
                "LANG": "ADQL",
                "FORMAT": "csv",
            },
            verify=cert_key,
        )

        post_sess_content = post_sess.content.decode()
        jobID = post_sess_content.split("CDATA[")[1].split("]]")[0]  # string not int

        # this is most likely not executing yet, so send the run to get the results
        # get the status and decode it
        post_sess_status = get_phase(jobID, session)
        if verbose:
            msgs.info(f"Post status run: {post_sess_status}")

        if post_sess_status == "PENDING":
            # send start request
            if verbose:
                msgs.info("Sending RUN phase.")

            session.post(
                f"https://easotf.esac.esa.int/tap-server/tap/async/{jobID}/phase",
                data={"phase": "RUN"},
                cookies=session.cookies,
            )

        executing = True
        stime = time.time()
        if verbose:
            msgs.info(f"Waiting for job completion. JobID: {jobID}")

        while executing:
            if get_phase(jobID, session) == "EXECUTING":
                time.sleep(5.0)
                if verbose and (int((time.time() - stime) % 20.0) == 0):
                    msgs.info(
                        "Waiting for query completion. "
                        f"Elapsed time {int((time.time() - stime)):d}s"
                    )

            elif get_phase(jobID, session) == "COMPLETED":
                executing = False
                if verbose:
                    msgs.info(f"Query completed in ~{int(time.time() - stime)}s.")
            else:
                raise ValueError(f"Unknown phase: {get_phase(jobID, session)}")

        # get the actual output
        post_sess_res = requests.get(
            f"https://easotf.esac.esa.int/tap-server/tap/async/{jobID}/results/result",
            cookies=session.cookies,
        ).content.decode()

        # write to output
        if verbose:
            msgs.info(f"Saving table to {savepath}")

        df = pd.read_csv(StringIO(post_sess_res))
        df.to_csv(savepath, index=False)


# =========================================================================== #


def download_table(query_table, user, savepath, sync=True, verbose=VERBOSE):
    savepath = Path(savepath)
    if not savepath.is_dir():
        raise ValueError(
            "Please provide the parent folder, table name is automatically generated!"
        )

    if not user.logged_in:
        msgs.info("User not logged in, trying log in.")
        user.sasotf_login()

    # minimal useful information (I think)
    query = select_query(query_table)

    # attach query type to the savepath to generate name
    savepath = savepath / f"{query_table}.csv"

    if sync:
        msgs.info(f"Downloading table `{query_table}` with a syncronous query.")
        # Try running in sync mode, otherwise default to async
        try:
            sync_query(query, user, savepath, cert_key=CERT_KEY, verbose=verbose)
        except (ChunkedEncodingError, ProtocolError):
            msgs.warn("Could not complete in sync mode, trying async query.")
            async_query(query, user, savepath, cert_key=CERT_KEY, verbose=verbose)

    # Async (already required for ivoa_obscore)
    else:
        msgs.info(f"Downloading table `{query_table}` with an asyncronous query.")

    # only for mosaic, only select Euclid product
    if query_table == "mosaic":
        return (
            pd.read_csv(savepath)
            .query("instrument_name == 'VIS' or instrument_name == 'NISP'")
            .reset_index(drop=True)
        )
    else:
        return pd.read_csv(savepath)


# =========================================================================== #


def parse_catalogue(tbl_in, inplace=False, force=False):
    if inplace:
        tbl = tbl_in
    else:
        tbl = tbl_in.copy()

    if "cutout_access_url" not in tbl.columns or force:
        msgs.info("Added `cutout_access_url` column.")
        # names are the same for both stack and calib, which is what we'll want to use most of the time
        access_urls = []
        # mosaic and calib/stack use a different identifier to download the images
        if "observation_id" in tbl.columns:
            for _p, _f, _o in zip(
                tbl["file_path"], tbl["file_name"], tbl["observation_id"]
            ):
                access_urls.append(build_access_url(_p.strip(), _f.strip(), obsid=_o))
        else:
            for _p, _f, _o in zip(
                tbl["file_path"], tbl["file_name"], tbl["tile_index"]
            ):
                access_urls.append(build_access_url(_p.strip(), _f.strip(), obsid=_o))

        tbl["cutout_access_url"] = access_urls

    # TODO: (Future us!) if this gets very slow (unlikely), merge the two loops
    if "image_access_url" not in tbl.columns or force:
        msgs.info("Added `image_access_url` column.")
        tbl["image_access_url"] = [
            f"https://easotf.esac.esa.int/sas-dd/data?file_name={_fn}&release=sedm&RETRIEVAL_TYPE=FILE"
            for _fn in tbl["file_name"]
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
    fname="", tbl_in=None, query_table="mosaic", user=None, use_local_tbl=False
):
    if user is None:
        user = User()

    if fname != "" and tbl_in is not None:
        msgs.warn(
            "Received both a table object and a path."
            "Ignoring table and redownloading."
        )

    # either read from file (and by default query and download it)
    # or directly pass a table
    if fname != "":
        if Path(fname).exists() and use_local_tbl:
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


def build_access_url(path, filename, obsid=None, tileidx=None):
    assert not (
        obsid is not None and tileidx is not None
    ), "Either `obsid` or `tileindex` must be None"
    # this builds the first part of the url
    # get collection from path itself
    collection = path.split("/")[-2]
    base = "https://easotf.esac.esa.int/sas-cutout/cutout"
    if obsid is None:
        params = f"filepath={path}/{filename}&collection={collection}&obsid={tileidx}"
    else:
        params = f"filepath={path}/{filename}&collection=MER&tileindex={obsid}"
    return f"{base}?{params}"


# =========================================================================== #


def build_url(access_url, ra, dec, side, search_type="CIRCLE"):
    # this adds the search parameter
    search = f"POS={search_type},{ra},{dec},{side}"
    return f"{access_url.strip()}&{search.strip()}"


# =========================================================================== #


@units.quantity_input()
def get_closest_image_url_local_tbl(
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
):
    # TODO: add possibility to use the query here
    # this takes the closes images to the target
    # TODO: Are these unique? Dithering bothers in this case
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

    return cat.iloc[inds], dist[inds]
    # return the full dataframe, pick out later
    #  whatever is needed


# =========================================================================== #


@units.quantity_input()
def get_closest_image_url_sas_otf(
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
):
    # FIXME!
    raise NotImplementedError


# =========================================================================== #


@units.quantity_input()
def get_closest_image_url(
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
    query_sas_otf=False,
):
    if query_sas_otf:
        return get_closest_image_url_sas_otf(
            ra, dec, cat, band, ra_cat=ra_cat, dec_cat=dec_cat
        )
    else:
        return get_closest_image_url_local_tbl(
            ra, dec, cat, band, ra_cat=ra_cat, dec_cat=dec_cat
        )


# =========================================================================== #


@units.quantity_input()
def get_download_urls(
    ra: units.deg, dec: units.deg, side: units.arcsec, cat, band, search_type="CIRCLE"
):
    side = side.to(units.deg).value
    image_urls = get_closest_image_url(ra, dec, cat, band)[0]["cutout_access_url"]
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


# =========================================================================== #
# Functions for persistence check
# =========================================================================== #


def create_persistence_qa_plot(
    ra, dec, result_df, cutout_dir, output_dir, mer_df=None, stack_df=None
):
    """Create a QA plot for the persistence check.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param result_df: DataFrame with the result data
    :type result_df: pandas.DataFrame
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param mer_df: DataFrame with the MER cutout data
    :type mer_df: pandas.DataFrame
    :param stack_df: DataFrame with the STACKED cutout data
    :type stack_df: pandas.DataFrame
    :return: None
    """

    # Reduce result_df to ra and dec in question with a delta of 1 arcsec
    delta_ra = 0.000278
    delta_dec = 0.000278
    df = result_df.query(
        "{} < ra < {} and {} < dec < {}".format(
            ra - delta_ra, ra + delta_ra, dec - delta_dec, dec + delta_dec
        )
    )
    df.query("processed == True and img_extension == img_extension", inplace=True)

    # Group result dataframe by observation id
    obs_id_gb = df.groupby("observation_id")

    for obs_id, df in obs_id_gb:
        plot_persistence_cutouts(
            ra,
            dec,
            obs_id,
            df,
            cutout_dir,
            output_dir,
            mer_df=mer_df,
            stack_df=stack_df,
        )


def plot_persistence_cutouts(
    ra, dec, obs_id, calib_df, cutout_dir, output_dir, mer_df=None, stack_df=None
):
    """Plot the cutouts of the persistence check.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param obs_id: Observation ID
    :type obs_id: int
    :param calib_df: DataFrame with the calibrated frame data
    :type calib_df: pandas.DataFrame
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param mer_df: DataFrame with the MER cutout data
    :type mer_df: pandas.DataFrame
    :param stack_df: DataFrame with the STACKED cutout data
    :type stack_df: pandas.DataFrame
    :return: None

    """

    # Plot setup
    cols = 4
    if mer_df is not None:
        cols += 1
    if stack_df is not None:
        cols += 1

    fig = plt.figure(figsize=(10, 2 * cols))
    fig.subplots_adjust(hspace=0.2)

    row_counter = 0

    # Add the MER mosaic cutouts
    # ToDo
    if mer_df is not None:
        row_counter += 1

    # Add the STACKED cutouts
    # ToDo
    if stack_df is not None:
        row_counter += 1

    # Add the NIR calibrated cutouts
    filter_dict = {"NIR_Y": 0, "NIR_J": 1, "NIR_H": 2}

    for _, row in calib_df.iterrows():
        filter_idx = filter_dict[row["filter_name"]]
        seq_idx = row["frame_seq"]
        filename = row["file_name"].replace(".fits", "_cutout.fits")

        file_path = os.path.join(cutout_dir, filename)

        cutout = whimg.Image(filename=file_path)

        ax_idx = seq_idx + filter_idx * 4 + 1
        ax = fig.add_subplot(3, cols, ax_idx)

        cutout._simple_plot(n_sigma=3, axis=ax, frameon=True, scalebar=None)

        cutout._add_aperture_circle(ax, ra, dec, 1)

        ax.set_title(
            "{} \n".format(row["filter_name"])
            + r" Flux {:.3f}+-{:.3f} $\mu$Jy".format(
                row["flux_aper_1.0"], row["flux_err_aper_1.0"]
            )
        )

        # Remove axis labels
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    # Save the plot
    plt.tight_layout()
    print(ra, dec)
    source_name = whut.coord_to_name([ra], [dec], epoch="J")[0]
    plt.savefig(
        os.path.join(
            output_dir, "{}_{}_persistence_cutouts.pdf".format(source_name, obs_id)
        )
    )


def check_persistence(
    ra,
    dec,
    calib_df,
    img_dir,
    cutout_dir,
    output_dir,
    cutout_fov=20,
    aperture_radii=np.array([1.0, 2.0]),
):
    """Check the persistence of a source in the Euclid images.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param calib_df: DataFrame with the calibrated frames
    :type calib_df: pandas.DataFrame
    :param img_dir: Path to the directory with the images
    :type img_dir: str
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param cutout_fov: Field of view of the cutout in arcsec. Defaults to 20 arcsec.
    :type cutout_fov: float
    :param aperture_radii: Radii of the apertures in arcsec. Defaults to [1., 2.]
    :type aperture_radii: numpy.ndarray
    :return: None

    """

    # Instantiate the SkyCoord object
    coord = SkyCoord(ra, dec, unit="deg", frame="icrs")

    # Downselect the calibrated frames to the NISP frames
    calib_columns = [
        "instrument_name",
        "observation_id",
        "pointing_id",
        "frame_seq",
        "filter_name",
        "file_name",
    ]

    result_df = calib_df.query('instrument_name == "NISP"')[calib_columns]
    result_df.loc[:, "ra"] = ra
    result_df.loc[:, "dec"] = dec
    result_df.loc[:, "processed"] = False
    result_df.loc[:, "img_extension"] = None

    for idx in result_df.index:
        file_path = os.path.join(img_dir, result_df.loc[idx, "file_name"])
        band = result_df.loc[idx, "filter_name"]
        survey = "Euclid-WIDE"

        # Test if the image exists
        if not os.path.exists(file_path):
            msgs.warn(f"File {file_path} does not exist.")
            continue
        else:
            msgs.info(f"Processing file {file_path}")
            result_df.loc[idx, "processed"] = True

        #  Cycle through extension to find the correct extension with the source
        hdul = fits.open(file_path)

        photfnu = hdul[0].header["PHOTFNU"]
        photrelex = hdul[0].header["PHRELEX"]

        for hdu in [h for h in hdul if "SCI" in h.name]:
            header = hdu.header
            wcs = WCS(header)

            # Test if the coordinate is within the image
            x, y = wcs.world_to_pixel(coord)

            # If source in extension, then
            if 0 < x < header["NAXIS1"] and 0 < y < header["NAXIS2"]:
                print("Coordinate is within the image")
                print("Extension: ", hdu.name)

                result_df.loc[idx, "img_extension"] = hdu.name

                # Load the image
                img = whimg.Image(file_path, exten=hdu.name, survey=survey, band=band)
                # Create a cutout image
                cutout = img.get_cutout_image(
                    coord.ra.value, coord.dec.value, cutout_fov
                )
                # Save the cutout image
                cutout_name = result_df.loc[idx, "file_name"].replace(
                    ".fits", "_cutout.fits"
                )
                cutout_path = os.path.join(cutout_dir, cutout_name)
                cutout.to_fits(cutout_path)

                # Calculate forced aperture photometry

                # nanomag_correction = 1 # ToDo: Implement the conversion to physical units
                zp_ab = img.header["ZPAB"]
                zp_ab_err = img.header["ZPABE"]
                gain = img.header["GAIN"]
                photreldt = img.header["PHRELDT"]

                print("ZPAB: ", zp_ab, band)

                nanomag_correction = np.power(10, 0.4 * (22.5 - zp_ab)) * gain

                phot_result = img.calculate_aperture_photometry(
                    coord.ra.value,
                    coord.dec.value,
                    nanomag_correction,
                    aperture_radii=aperture_radii,
                    exptime_norm=1,
                    background_aperture=np.array([7, 10.0]),
                )

                # Create a metric that flags persistence
                # ToDo!!!

                # Save the results to the result DataFrame
                prefix = "{}_{}".format(survey, band)
                for radius in aperture_radii:
                    raw_flux = phot_result[
                        "{}_raw_aper_sum_{:.1f}arcsec".format(prefix, radius)
                    ]
                    raw_flux_err = phot_result[
                        "{}_raw_aper_sum_err_{:.1f}arcsec".format(prefix, radius)
                    ]
                    # flux = phot_result['{}_flux_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # flux_err = phot_result['{}_flux_err_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # snr = phot_result['{}_snr_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # abmag = phot_result['{}_mag_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # abmag_err = phot_result['{}_mag_err_aper_{:.1f}arcsec'.format(prefix, radius)]

                    # Calculation according to
                    # https://apceuclidccweb-pp.in2p3.fr/Documentation/NIR/NIR_AbsolutePhotometry/develop/0.5.0/md_README.html

                    flux = (
                        raw_flux / 87.2248 * photfnu * photrelex * photreldt
                    )  # flux in micro Jy
                    flux_err = raw_flux_err / 87.2248 * photfnu * photrelex * photreldt
                    snr = flux / flux_err

                    abmag = -2.5 * np.log10(raw_flux) + zp_ab

                    result_df.loc[idx, "raw_aper_sum_{:.1f}".format(radius)] = raw_flux
                    result_df.loc[idx, "flux_aper_{:.1f}".format(radius)] = flux
                    result_df.loc[idx, "flux_err_aper_{:.1f}".format(radius)] = flux_err
                    result_df.loc[idx, "snr_aper_{:.1f}".format(radius)] = snr
                    result_df.loc[idx, "abmag_aper_{:.1f}".format(radius)] = abmag
                    result_df.loc[idx, "abmag_err_aper_{:.1f}".format(radius)] = (
                        zp_ab_err
                    )

    # Create the persistence plot
    create_persistence_qa_plot(ra, dec, result_df, cutout_dir, output_dir)

    coord_name = whut.coord_to_name([ra], [dec], epoch="J")[0]
    table_name = "{}_persistence_check.csv".format(coord_name)
    result_df.to_csv(os.path.join(output_dir, table_name), index=False)


# Have a dictionary of filepath that links to the correct cutout files for the persistence check


def prepare_sas_catalogue(
    cat_path,
    sas_catalogue,
    user,
    product_type,
    product_type_dict,
    use_local_tbl=False,  # default in any case, better make it explicit
):
    cat = load_catalogue(
        fname=cat_path,
        query_table=sas_catalogue,
        user=user,
        use_local_tbl=use_local_tbl,
    )[0]

    # select only the products that you want
    # parse_catalogue just does some renaming to make sure that
    # different tables can be handles with the same code
    parse_catalogue(cat, inplace=True)

    cat_data_prod = cat.query(
        f"product_type == '{product_type_dict[product_type][0]}' "
        + f"or product_type == '{product_type_dict[product_type][1]}'"
    ).reset_index()

    if cat_data_prod.empty:
        msgs.error(
            f"No data product of type {product_type} found. Are you using the correct table?"
        )
        raise ValueError
    else:
        return cat_data_prod


# =========================================================================== #


# sourced from: https://github.com/tqdm/tqdm/#hooks-and-callbacks, requests version
def download_with_progress_bar(url, user, out_fname):
    response = requests.get(url, cookies=user.cookies, stream=True)
    with tqdm.wrapattr(
        open(out_fname, "wb"),
        "write",
        miniters=1,
        desc=msgs.info(f"Downloading {url.split('=')[1].split('&')[0]}"),
        total=int(response.headers.get("content-length", 0)),
    ) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


# =========================================================================== #


@units.quantity_input()
def download_full_image(
    ra: units.deg,
    dec: units.deg,
    user,
    cat_outpath,
    img_outpath,
    img_outname=None,
    img_type="calib",
    verbose=False,
):
    """Download a full EUCLID image given an url. Note that in this case the
    `img_type` indicates both the table to query and the data product used,
    as the distinction is only used for ivoa_obscore.

    :param url: URL of the image to download.
    :type url: str
    :param image_name: Name of the image to download.
    :type image_name: str
    :return:
    """
    cat_outpath = Path(cat_outpath)
    img_outpath = Path(img_outpath)

    # Restric image type
    if img_type not in ["mosaic", "calib"]:
        raise ValueError("`img_type` should be either 'mosaic' or 'calib'")

    # download new tile catalogue and process it
    tile_cat = prepare_sas_catalogue(
        cat_outpath,
        img_type,
        user,
        img_type,
        product_type_dict,
        use_local_tbl=False,
    )

    # download all images - the logic behind this is to group by identifier
    #  to minimize the number of images to download
    df = None

    for ra_, dec_ in zip(ra, dec):
        partial = get_closest_image_url(
            ra_,
            dec_,
            tile_cat,
            "[VIS, Y, J, H]",
            query_sas_otf=False,
        )[0]

        df = partial if df is None else pd.concat([df, partial], ignore_index=True)

    # this gets all 4 bands at the same time
    unique_identifier = (
        "mosaic_product_oid" if img_type == "mosaic" else "calibrated_frame_oid"
    )
    df_urls = df.drop_duplicates(unique_identifier, ignore_index=True)

    # actually download the images
    for _, row in df_urls.iterrows():
        if img_outname is None:
            img_outname = row["file_name"]

        try:
            download_with_progress_bar(
                row["image_access_url"], user, img_outpath / img_outname
            )

            if verbose > 0:
                msgs.info(f"Download of {img_outname} to {img_outpath} completed")

        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            msgs.warn(f"Download error encountered: {err}")
            if verbose > 0:
                msgs.warn(f"Download of {img_outname} unsuccessful")

    # and save the catalogue just in case it is needed for anything
    df_urls.to_csv(img_outpath / "Euclid_urls_vetted.csv")
