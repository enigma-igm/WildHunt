

import pandas as pd
import wildhunt.catalog as whcat
import wildhunt.image as whim
from wildhunt import plotting
from wildhunt import utils

if __name__ == '__main__':

    ra_3 = utils.convert_hmsra2decdeg('13:50:23.60')
    dec_3 = utils.convert_dmsdec2decdeg('+37:48:35.72')
    print(ra_3, dec_3)
    # # Create a dataframe
    ra = [210.336917, 230.877792, ra_3]
    dec = [45.71485, 29.594353, dec_3]
    id = ['J1401', 'J1523', 'J1350']

    data = {'id': id,
            'ra': ra,
            'dec': dec}

    df = pd.DataFrame(data)

    # Create a catalog
    cat = whcat.Catalog('lofar_quasars',
                        'ra', 'dec', 'id',
                        table_data=df)
    #
    # # Set up survey dictionary and retrieve imaging data
    survey_dict = [
        # {'survey': 'PS1', 'bands': ['g', 'r', 'i', 'z'], 'fov': 50},
        {'survey': 'DELSDR9', 'bands': ['g', 'r', 'z'],  'fov': 120},
        {'survey': 'UHSDR1', 'bands': ['J'], 'fov': 120},
    ]

    cat.get_survey_images('cutouts', survey_dict)


    # filename = 'cutouts/J152330.67+293539.67_UKIDSSDR11PLUSLAS_J_fov120.fits'
    # img = whim.Image(230.877792, 29.594353,'UKIDSSDR11PLUSLAS', 'J', 'cutouts',
    # 50)
    #
    # img.show(10)

    plotting.plot_source_images(230.877792, 29.594353, survey_dict, 50)




