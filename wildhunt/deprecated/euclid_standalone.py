import os
from pathlib import Path

from astropy import units

from wildhunt import euclid_utils as eu

LOCAL_PATH = Path(os.environ.get("WILDHUNT_LOCALPATH"))

# FRA, DO NOT CHANGE THIS LINE UNTIL YOU ARE SURE YOU KNOW WHAT YOU ARE DOING. DON'T!
# https://easotf.esac.esa.int/sas-cutout/cutout?filepath=/data_03/repository_otf/NIR/65879/EUC_NIR_W-CAL-IMAGE_H-65879-39_20240130T023214.787691Z.fits&collection=NIR&obsid=65879&POS=CIRCLE,149.7848750,2.0673917,0.0033

if __name__ == "__main__":
    # TODO: Update this after the latest changes
    
    # I know this object is there, has this coords and is in a EUCLID image see link above
    # 79 - 51 is a random field, to be used for stack images
    # 149 2 is a random galaxy, to be used for calibrated images
    obj_ra, obj_dec, side = (
        79.651702 * units.deg,  # 79.651702 * units.deg,  # 149.7848750 * units.deg,
        -51.935266 * units.deg,  # -51.935266 * units.deg, # 2.0673917 * units.deg,
        10.0 * units.arcsec,
    )

    user = eu.User()
    user.sasotf_login()

    cat = eu.load_catalogue(
        fname=LOCAL_PATH / "ivoa_frames.csv",
        query_table="ivoa_obscore",
        user=user,
        overwrite=False,
    )[0]

    eu.parse_catalogue(cat, inplace=True)

    cat_stack = cat.query("product_type == 'DpdNirStackedFrame' or product_type == 'DpdVisStackedFrame'").reset_index()
    urls_stack = eu.get_download_urls(obj_ra, obj_dec, side, cat_stack, ['VIS', 'J', 'H'])

    # We want the stack but calibrated images seem to be more abundant
    # TODO: Figure out the difference between the two!
    # cat_nir_calib = cat.query("product_type == 'DpdNirCalibratedFrame'").reset_index()
    # cat_vis_calib = cat.query("product_type == 'DpdVisCalibratedFrame'").reset_index()
    # urls_nir_calib = get_download_urls(obj_ra, obj_dec, side, cat_nir_calib)
    # urls_vis_calib = get_download_urls(obj_ra, obj_dec, side, cat_vis_calib)

    eu.download_cutouts(
        obj_ra, obj_dec, urls_stack, LOCAL_PATH / "test_images", user=user
    )