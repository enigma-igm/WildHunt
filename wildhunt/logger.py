import logging
import sys

# Better logging and verbosing

# Logger: https://stackoverflow.com/a/48891485
# and this: https://stackoverflow.com/a/66062313

# This will not correctly set the child for any of the file in the .wildhunt
#  folder, which might, ot might not be what we want
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S"
)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
