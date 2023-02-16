#!/usr/bin/env python

import time

from wildhunt import image, catalog, plotting

from IPython import embed

if __name__ == "__main__":

    t0 = time.time()

    cat = catalog.Catalog('example', 'RA', 'DEC', 'Name', datapath='UKIDSS_sources.csv')

    survey_dict = [
                {'survey': 'PS1', 'bands': ['i', 'z', 'y'], 'fov':60},
               {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['Y', 'J', 'H', 'K'], 'fov':60},
               {'survey': 'DELSDR9', 'bands': ['g', 'r', 'z', '1', '2'], 'fov':60}
                ]

    ra = cat.df.compute()[cat.ra_colname][:10]
    dec = cat.df.compute()[cat.dec_colname][:10]
    plotting.generate_cutout_images(ra, dec, survey_dict, n_jobs=10)

    print("{:.1f} s: ".format( time.time() - t0))