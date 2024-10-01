from pathlib import Path

import pandas as pd

from wildhunt import euclid_utils as eu

user = eu.User()
user.sasotf_login()

manual = False

prefix = Path.home()

if __name__ == "__main__":
    if manual:
        # if one has all the input to the function available
        ra = 174.61753
        dec = 72.4415851
        calib_df = pd.read_csv("../examples/data/20240925_persistence_testing.csv")
        img_dir = "/Users/jtschindler/Downloads/"
        cutout_dir = "../examples/cutouts/"
        output_dir = "."

        eu.check_persistence(ra, dec, calib_df, img_dir, cutout_dir, ".")
    else:
        # complete pipeline:
        # from the coordinate downloads the necessary images,
        # sets up the folders and produces the plots
        img_folder = prefix / "persistence_check" / "images"
        cutout_folder = prefix / "persistence_check" / "cutout"
        output_folder = prefix / "persistence_check" / "output"

        for dir_ in [img_folder, cutout_folder, output_folder]:
            if not dir_.exists():
                dir_.mkdir(parents=True, exist_ok=True)

        eu.persistance_pipeline(
            [174.61753],
            [72.4415851],
            user,
            img_folder,
            cutout_folder,
            output_folder,
            # download_function=eu.download_esa_datalab,
            # Comment ^ this in if running on the esa datalab
        )
