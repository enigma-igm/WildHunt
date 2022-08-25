
import pandas as pd

from wildhunt import findingchart as fc
from wildhunt import utils

def single_chart_example():

    df = pd.read_csv('LRIS_VI_candidates_220304.csv')

    ra = df.iloc[0]['RA']
    dec = df.iloc[0]['DEC']


    survey = 'DELSDR9'
    band = 'z'
    aperture = 2
    fov = 120

    image_folder_path = './cutouts'

    offsets = pd.read_csv('offset_stars.csv')

    fig = fc.make_finding_chart(ra, dec, survey, band, aperture, fov,
                           image_folder_path,
                           offset_df=offsets,
                           offset_id=0,
                           offset_focus=False,
                           offset_ra_column_name='offset_ra',
                           offset_dec_column_name='offset_dec',
                           offset_mag_column_name='mag_z',
                           offset_id_column_name='offset_shortname',
                           label_position='bottom',
                           slit_width=None, slit_length=None,
                           position_angle=0, verbosity=0)


    fig.save('fc_{}.pdf'.format('test'), transparent=False)



def charts_from_table_example():


    df = pd.read_csv('UKIDSS_sources.csv')

    ra_column_name = 'RA'
    dec_column_name = 'DEC'
    target_column_name = 'Name'
    survey = 'UKIDSSDR11PLUSLAS'
    band = 'J'
    aperture = 2
    fov = 50
    image_folder_path = './cutouts'

    fc.make_finding_charts(df, ra_column_name, dec_column_name,
                        target_column_name, survey, band,
                        aperture, fov, image_folder_path,
                           auto_download=False)




def new_charts_from_table_example():


    cat = Catalog(filename, ra_colname, dec_colnam, id_colname)

    cat.make_finding_charts(target_colname, survey, band, aperture ...)

single_chart_example()
# charts_from_table_example()