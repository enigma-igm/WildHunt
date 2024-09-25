
import pandas as pd

from wildhunt import euclid_utils as wheuclid


if __name__ == '__main__':

    ra = 174.61753
    dec = 72.4415851
    calib_df = pd.read_csv('../examples/data/20240925_persistence_testing.csv')
    img_dir = '/Users/jtschindler/Downloads/'
    cutout_dir = '../examples/cutouts/'
    output_dir = '.'

    wheuclid.check_persistence(ra, dec, calib_df, img_dir, cutout_dir, '.')

