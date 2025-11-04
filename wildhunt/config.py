import os

from wildhunt import pypmsgs

msgs = pypmsgs.Messages()

EUCLID_ENV = os.getenv("WILDHUNT_EUCLID_ENV", "IDR")

def set_euclid_env(env: str):
    """Set the application's environment by updating the module-level EUCLID_ENV variable.

    :param env: The environment identifier to set ('IDR', 'OTF', 'REG', not case-sensitive).
    :type env: str
    :return: None
    :rtype: None
    """
    global EUCLID_ENV
    EUCLID_ENV = env.lower()

    msgs.info(f"Euclid environment set to: {EUCLID_ENV.upper()}")

