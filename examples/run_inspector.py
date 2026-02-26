
import pandas as pd

from wildhunt import inspector


if __name__ == '__main__':
    # Read in the candidate catalog as a pandas dataframe
    my_candidate_df = pd.read_csv('./data/pselqs_quasars_subset.csv')

    # Specify RA and Dec column names
    my_ra_column_name = 'ps_ra'
    my_dec_column_name = 'ps_dec'

    # Field of view of cutout images in arcseconds
    fov = 10

    # Set the directory name for the cutout images
    my_cutout_dir = 'cutouts'

    surveys = ['PS1', 'PS1', 'PS1', 'PS1']
    bands = ['g', 'r', 'i', 'z']

    # ra = my_candidate_df.loc[10, my_ra_column_name]
    # dec = my_candidate_df.loc[10, my_dec_column_name]

    # Set field of views of the
    # fovs = [5, 5, 5, 5]
    # apertures = [0.5, 0.5, 0.5, 0.5]
    # square_sizes = [2, 2, 2, 2]

    # Set visual classes for "button" classification
    visual_classes = ['good', 'edge', 'artifact', 'blend']

    # add_info_list = [('column', 'Milliquas citation', 'mq_cite')]
    add_info_list = []

    # List of magnitude column names, list with length N
    mag_column_names = []
    # List of magnitude error column names, list with length N
    magerr_column_names = []

    # Run a simple example
    inspector.run(my_candidate_df, my_cutout_dir, my_ra_column_name,
                  my_dec_column_name, surveys, bands,
                  # mag_column_names=mag_column_names,
                  # magerr_column_names=magerr_column_names,
                  # add_info_list=add_info_list,
                  minimum_fov=10,
                  visual_classes=visual_classes, verbosity=2)