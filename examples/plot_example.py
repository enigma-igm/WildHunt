

import pandas as pd
import wildhunt.catalog as whcat
import wildhunt.image as whim


if __name__ == '__main__':

    # # Create a dataframe
    # ra = [210.336917, 230.877792]
    # dec = [45.71485, 29.594353]
    # id = ['J1350', 'J1523']
    #
    # data = {'id': id,
    #         'ra': ra,
    #         'dec': dec}
    #
    # df = pd.DataFrame(data)
    #
    # # Create a catalog
    # cat = whcat.Catalog('lofar_quasars',
    #                     'ra', 'dec', 'id',
    #                     table_data=df)
    #
    # # Set up survey dictionary and retrieve imaging data
    # survey_dict = [
    #     # {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z', 'y'], 'fov': 50},
    #     {'survey': 'UKIDSSDR11PLUSLAS', 'bands': ['J'], 'fov': 120},
    #     # {'survey': 'DELS', 'bands': ['g', 'r', 'z'],  'fov': 120}
    # ]
    #
    # cat.get_survey_images('cutouts', survey_dict)


    filename = 'cutouts/J152330.67+293539.67_UKIDSSDR11PLUSLAS_J_fov120.fits'
    img = whim.Image(230.877792, 29.594353,'UKIDSSDR11PLUSLAS', 'J', 'cutouts',
    50)

    img.show(10)




