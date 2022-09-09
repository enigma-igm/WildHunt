#!/usr/bin/env python

import time

from wildhunt import image, catalog

from IPython import embed

if __name__ == "__main__":

    t0 = time.time()

    cat = catalog.Catalog('example', 'RA', 'DEC', 'Name', datapath='UKIDSS_sources.csv')

    survey_dict = [
                {'survey': 'PS1', 'bands': ['g', 'r'], 'fov':50},
               {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'], 'fov':120},
               {'survey': 'DELSDR9', 'bands': ['z', '1'], 'fov':120}
                ]

    cat.get_survey_images('cutouts',  survey_dict, n_jobs=10)

    cat.get_forced_photometry(survey_dict, 'test', n_jobs=10)

    #ra = cat.df.compute()[cat.ra_colname]
    #dec = cat.df.compute()[cat.dec_colname]
    #image.forced_photometry(ra, dec, survey_dict, 'test', radii=[1.], n_jobs=10)

    print("{:.1f} s: ".format( time.time() - t0))