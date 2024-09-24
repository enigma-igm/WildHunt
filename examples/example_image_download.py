#!/usr/bin/env python

"""
A simple example showing how to download images from the online image servers
implemented in wildhunt.
"""

from wildhunt import catalog

if __name__ == "__main__":

    # Instantiate the catalog class, loading the pselqs_quasars.csv file
    # from the data folder.
    cat = catalog.Catalog('example', 'ps_ra', 'ps_dec', 'wise_designation',
                          datapath='./data/pselqs_quasars_subset.csv')

    # Definding a list of dictionaries containing the survey information.
    fov = 120  # field of view in arcseconds

    survey_dict = [
        # {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov':fov},
        {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'], 'fov':fov},
        # {'survey': 'DELSDR10', 'bands': ['z'], 'fov': fov},
        # {'survey': 'allWISE', 'bands': ['3', '4'], 'fov': fov},
        {'survey': 'LoTSSDR2', 'bands': ['150MHz', '150MHz_lowres'], 'fov': fov},
    ]

    # Download the images from the surveys defined in the survey_dict
    # The images are downloaded into the folder 'cutouts' in the current
    # working directory.
    cat.get_survey_images('./cutouts',  survey_dict, n_jobs=5)
