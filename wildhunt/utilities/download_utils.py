#!/usr/bin/env python
import os
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from wildhunt import pypmsgs
from wildhunt.utilities import queries
from wildhunt.utilities import query_utils as whqu

# =========================================================================== #


msgs = pypmsgs.Messages()

ChunkedEncodingError = requests.exceptions.ChunkedEncodingError
ProtocolError = requests.urllib3.exceptions.ProtocolError


# =========================================================================== #


if os.environ.get("WILDHUNT_LOCALPATH") is None:
    LOCAL_PATH = Path.home()
else:
    LOCAL_PATH = Path(os.environ.get("WILDHUNT_LOCALPATH"))

CERT_KEY = LOCAL_PATH / "eas-esac-esa-int-chain.pem"
if not CERT_KEY.exists():
    CERT_KEY = False
VERBOSE = 0


# =========================================================================== #
# ========== Functions to download large files (e.g., full imgages) ========= #
# =========================================================================== #


# sourced from: https://github.com/tqdm/tqdm/#hooks-and-callbacks, requests version
# TODO: Add a check for empty content, somehow
def download_with_progress_bar(url, user, out_fname, check_for_existing=True):
    """Download a file from the given URL and display a progress bar during the download.

    This function retrieves a file from a specified URL using the given user credentials
    and saves it to the specified output file path. If the file already exists, it skips
    the download and uses the cached version instead. The download process is displayed
    with a progress bar for better visualization of the download status.

    :param url: The URL from which to download the file.
    :type url: str
    :param user: The user object for authentication; this is used to manage cookies for the request.
    :type user: User
    :param out_fname: The path where the downloaded file will be saved.
    :type out_fname: pathlib.Path
    :return: None; downloads the file and saves it to the specified location.
    """
    if check_for_existing and out_fname.exists():
        # TODO: How does this work with corrupted files?
        # answer: it does not work, there should be some kind of logic to check this
        msgs.info(f"File {out_fname} already exists, using cached version.")
        return

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


def download_without_progress_bar(url, user, out_fname, check_for_existing=True):
    """Download a file from the given URL without displaying a progress bar.

    This function retrieves a file from the specified URL using the provided user credentials
    and saves it to the intended output file path. If the file already exists, it uses the cached
    version and skips the download. If the download attempt is unsuccessful, it issues a warning.

    :param url: The URL from which to download the file.
    :type url: str
    :param user: The user object for authentication; this is used to manage cookies for the request.
    :type user: User
    :param out_fname: The path where the downloaded file will be saved.
    :type out_fname: pathlib.Path
    :return: None; attempts to download the file and save it to the specified location.
    """
    if check_for_existing and out_fname.exists():
        # TODO: How does this work with corrupted files?
        # answer: it does not work, there should be some kind of logic to check this
        msgs.info(f"File {out_fname} already exists, using cached version.")
        return

    response = requests.get(url, cookies=user.cookies, stream=True)

    # if there is actual content in the image I downloaded
    if len(response.content) == 0:
        msgs.warn(f"Empty fits content for {out_fname}!")

    if response.status_code == 200:
        with open(out_fname, "wb") as fout:
            fout.write(response.content)
    else:
        msgs.warn(f"Download of {out_fname} failed.")


# =========================================================================== #


def download_esa_datalab(url, user, out_fname):
    # same interface, but the parameters required are slightly different
    #  so some parsing is required
    msgs.warn(
        "This will only work on the ESA Datalab "
        "until the astroquery Euclid package is available!"
    )
    from astroquery.esa.euclid.core import Euclid  # type: ignore

    # would be useseful a simple Euclid.is_logged_in
    # Anyway:
    if not Euclid._EuclidClass__eucliddata._TapPlus__isLoggedIn:
        Euclid.login(user=user.username)

    target_file_name = url.split("file_name=")[-1].split("&")[0]
    Euclid.get_product(file_name=target_file_name, output_file=out_fname)


# =========================================================================== #
# =========== Functions to download small items (e.g., catalogues) ========== #
# =========================================================================== #


def download_full_table_from_sas(
    query_table, user, savepath, sync_query=True, verbose=VERBOSE
):
    """Download a specified table from the EUCLID TAP server and save it as a CSV file.

    This function checks if the user is logged in, constructs an appropriate query, and
    attempts to download the specified table either synchronously or asynchronously.
    If the query table is 'mosaic', it filters the results to include only EUCLID related
    data products.

    :param query_table: The name of the table to download (e.g., 'mosaic', 'stack', etc.).
    :type query_table: str
    :param user: The user object containing session cookies for authentication.
    :type user: User
    :param savepath: The path to the directory where the CSV file will be saved.
    :type savepath: str or Path
    :param sync: If True, use synchronous query mode; otherwise, use asynchronous.
    :type sync: bool
    :param verbose: If greater than 0, enables verbose output for logging.
    :type verbose: int
    :raises ValueError: If the savepath is not a directory or if the user is not logged in.
    :raises IOError: If there is an error reading the downloaded CSV file.
    :return: A pandas DataFrame containing the downloaded table data, potentially filtered.
    :rtype: pandas.DataFrame
    """
    savepath = Path(savepath)
    if not savepath.is_dir():
        raise ValueError(
            "Please provide the parent folder, table name is automatically generated!"
        )

    # check if you are logged in. If not, do the login
    user._check_for_login()

    # minimal useful information (I think)
    query = queries.query_full_table(query_table)

    # attach which table I am querying to the savepath to generate name
    savepath = savepath / f"{query_table}.csv"

    if sync_query:
        msgs.info(
            f"Downloading table `{query_table}` with a syncronous query to {savepath}."
        )
        # Try running in sync mode, otherwise default to async
        try:
            whqu.sync_query(query, user, savepath, cert_key=CERT_KEY, verbose=verbose)
        except (ChunkedEncodingError, ProtocolError):
            msgs.warn("Could not complete sync query, trying with async.")
            whqu.async_query(query, user, savepath, cert_key=CERT_KEY, verbose=verbose)

    # Async (already required for ivoa_obscore)
    else:
        msgs.info(
            f"Downloading table `{query_table}` with an asyncronous query to {savepath}."
        )

    # only for mosaic, only select Euclid product
    #  mosaic includes many many many more things from other surveys
    if query_table == "mosaic":
        return (
            pd.read_csv(savepath)
            .query("instrument_name == 'VIS' or instrument_name == 'NISP'")
            .reset_index(drop=True)
        )
    else:
        return pd.read_csv(savepath)
