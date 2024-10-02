#!/usr/bin/env python
import getpass
import os
from http.cookiejar import MozillaCookieJar
from pathlib import Path

import requests

from wildhunt import pypmsgs
from wildhunt.utilities.general_utils import b64d, b64e

msgs = pypmsgs.Messages()

# =========================================================================== #


if os.environ.get("WILDHUNT_LOCALPATH") is None:
    LOCAL_PATH = Path.home()
else:
    LOCAL_PATH = Path(os.environ.get("WILDHUNT_LOCALPATH"))

CERT_KEY = LOCAL_PATH / "eas-esac-esa-int-chain.pem"
if not CERT_KEY.exists():
    CERT_KEY = False

# =========================================================================== #

VERBOSE = 0

# =========================================================================== #


class User(object):
    """Basic user class to handle login for accessing the EUCLID OTF archive.

    The password is encoded in Base64 to prevent accidental printing to screen.
    User data can be loaded from a configuration file.

    :param username: The username for login. If None, will prompt the user for input.
    :type username: str or None
    :param password: The password for login. If None, will prompt the user for input.
    :type password: str or None
    :param filepath: The path to the configuration file containing user data.
    :type filepath: Path
    :param encoded: Whether the password is already encoded in Base64. Default is True.
    :type encoded: bool
    """

    def __init__(
        self,
        username=None,
        password=None,
        filepath=LOCAL_PATH / "sas_otf_user_data.cfg",
        encoded=True,
    ):
        """Initialize the User object, loading user data from the configuration file if it exists.

        :param username: Username for login.
        :param password: Password for login.
        :param filepath: Path to user data configuration file.
        :param encoded: If True, the provided password should be considered already encoded.
        """
        # check whether we have a configuration file. If so, try to load the data from there
        if filepath.exists() or (username is not None and password is not None):
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
        """Load user data from the specified configuration file.

        :param filepath: Path to the configuration file.
        :type filepath: Path
        """
        try:
            with open(filepath, "r") as f:
                data = f.read().split()
                self.username = data[0]
                self.password = b64e(data[1])
        except FileNotFoundError:
            msgs.info(f"{filepath} not found!")

    # ======================================================================= #

    def set_user_data(self, force=False):
        """Prompt to set or reset the user data (username and password).

        If `force` is True, the user will be prompted regardless of existing data.

        :param force: If True, force the prompt for new username and password.
        :type force: bool
        """
        if self.username is None or force:
            self.username = input("Enter user name: ")
        else:
            msgs.info("User already set, pass `force` to reset it.")

        if self.password is None or force:
            self.password = b64e(getpass.getpass("Enter password: "))
        else:
            msgs.info("Password already set, pass `force` to reset it.")

        # TODO: Check the encoding logic here
        self.login_data = {"username": self.username, "password": self.password}

    # ======================================================================= #

    def get_user_data(self):
        """Retrieve the user's login data as a dictionary.

        :return: A dictionary containing the username and decoded password.
        :rtype: dict
        """
        return {
            "username": self.login_data["username"],
            "password": b64d(self.login_data["password"]),
        }

    # ======================================================================= #

    def store_user_data(
        self, path=LOCAL_PATH, fname="sas_otf_user_data.cfg", overwrite=False
    ):
        """Store the user data to a configuration file.

        If the file already exists, the user is prompted to confirm whether to
        overwrite it. If the `overwrite` parameter is set to True, the file will
        be overwritten without prompting the user.

        :param path: Path to the directory where the configuration file will be saved.
        :type path: Path
        :param fname: Name of the configuration file to save the user data.
        :type fname: str
        :param overwrite: If True, overwrite the existing file without confirmation.
        :type overwrite: bool
        """
        outfile = path / fname
        if outfile.exists() and not overwrite:
            user_overwrite = input("[Info] Table exists, overwrite? [y]/n ").lower()
            if user_overwrite in ["y", "\n"]:
                overwrite = True

        if not outfile.exists() or overwrite:
            with open(outfile, "w") as f:
                f.write(f"{self.username} {self.password}")

    # ======================================================================= #

    def sasotf_login(self, cert_key=CERT_KEY):
        """Log in to the EUCLID OTF archive using the stored user credentials.

        This function sends login requests to multiple services of the EUCLID OTF server.

        :param cert_key: Certificate key for verifying the server connection.
        :type cert_key: str
        """
        if self.login_data is None:
            self.set_user_data()

        cookies = MozillaCookieJar()
        with requests.Session() as session:
            session.cookies = cookies

            # These are all needed for the different services that we use
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

    def _check_for_login(self):
        if not self.logged_in:
            msgs.info("User not logged in, trying log in.")
            self.sasotf_login()

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
        # in this case don't really care about the extra info - I just need to know the current user and if I am (possibly)
        #   logged in
        out = f"euclid_download.User()\nCurrent user: {self.username}\n"
        if self.logged_in:
            out += "User logged in. If experiencing problems, reload the user data and/or try logging in again."
        else:
            out += "User not logged in."

        return out
