#!/usr/bin/env python
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from wildhunt import pypmsgs

msgs = pypmsgs.Messages()


# TODO: Use raise_for_status instead of manually handling the request error

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


def get_phase(jobID, session):
    """Retrieve the processing phase of a running job using its job ID.

    This function sends a GET request to the EUCLID OTF TAP server to obtain
    the current phase of the specified asynchronous job. The request uses the
    provided session's cookies for authentication.

    :param jobID: The ID of the job whose phase is to be retrieved.
    :type jobID: str
    :param session: The requests session containing cookies for authentication.
    :type session: requests.Session
    :return: The current phase of the job as a string.
    :rtype: str
    """
    return requests.get(
        f"https://easotf.esac.esa.int/tap-server/tap/async/{jobID}/phase",
        cookies=session.cookies,
    ).content.decode()


# =========================================================================== #


def async_query(query, user, savepath, cert_key=CERT_KEY, verbose=VERBOSE):
    """Execute an asynchronous SQL query against the EUCLID TAP server and save the results to a file.

    This function sends an asynchronous query request to the EUCLID OTF TAP server,
    waits for the query to complete, and retrieves the results in CSV format. The results
    are then saved to the specified file path.

    :param query: The SQL query to execute in ADQL format.
    :type query: str
    :param user: The user object containing session cookies for authentication.
    :type user: User
    :param savepath: The path where the results should be saved.
    :type savepath: str
    :param cert_key: The certificate key for verifying server connections (optional).
    :type cert_key: str
    :param verbose: If greater than 0, enables verbose output for logging.
    :type verbose: int
    :return: A pandas DataFrame containing the query results.
    :rtype: pandas.DataFrame
    :raises ValueError: If the job status is unknown or if any request fails.
    :raises IOError: If there is an error writing the results to the specified file.
    """
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
        return df


# =========================================================================== #


def sync_query(query, user, savepath, cert_key=CERT_KEY, verbose=VERBOSE):
    """Execute a synchronous SQL query against the EUCLID TAP server and save the results to a file.

    This function sends a GET request to the EUCLID OTF TAP server to execute
    the provided SQL query and retrieves the results in CSV format. If the query
    is successful, the results are saved to the specified file path.

    :param query: The SQL query to execute in ADQL format.
    :type query: str
    :param user: The user object containing session cookies for authentication.
    :type user: User
    :param savepath: The path where the results should be saved.
    :type savepath: str
    :param cert_key: The certificate key for verifying server connections (optional).
    :type cert_key: str
    :param verbose: If greater than 0, enables verbose output for logging.
    :type verbose: int
    :raises ValueError: If the server responds with a status code other than 200.
    :raises IOError: If there is an error writing the results to the specified file.
    """
    response = requests.get(
        "https://easotf.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
        + query.replace(" ", "+"),
        verify=cert_key,
        cookies=user.cookies,
    )

    if verbose > 0:
        msgs.info(f"Response code: {response.status_code}")

    if response.status_code == 200 and savepath is not None:
        if verbose:
            msgs.info(f"Successfully retrieved table, writing to {savepath}.")

        content = response.content.decode("utf-8")
        with open(savepath, "w") as f:
            if verbose:
                msgs.info(f"Saving table to {savepath}")
            f.write(content)

        return content
    elif response.status_code == 200 and savepath is None:
        content = response.content.decode("utf-8")
        return content
    else:
        msgs.error(
            f"Something went wrong. Query returned status code {response.status_code}"
        )
        raise ValueError("Failed query, please retry.")


# =========================================================================== #
