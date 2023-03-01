#!/usr/bin/env python

import time
from wildhunt import catalog


if __name__ == "__main__":

    t0 = time.time()

    cat = catalog.Catalog('example', 'RA', 'DEC','Name',
                          datapath='UKIDSS_sources.csv')

    survey_dict = [
        {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':50},
         {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'],
          'fov':120},
        {'survey': 'DELSDR9', 'bands': ['z'],
         'fov': 120},
        {'survey': 'allWISE', 'bands': ['3', '4'], 'fov': 120}
    ]

    cat.get_survey_images('cutouts',  survey_dict, n_jobs=3)
    print("{:.1f} s: ".format(time.time() - t0))


