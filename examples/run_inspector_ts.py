import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

from wildhunt import inspector_ts
from wildhunt.utils import coord_to_name


def relink_to_expected_names(df, ra_col, dec_col, cutout_dir, surveys, bands):
    """Symlink cutout files named after 'euclid_designation' to the name
    coord_to_name() derives from ra/dec, since the two naming conventions
    disagree on the number of decimal digits used for the Dec arcseconds.
    """
    for _, row in df.iterrows():
        real_name = row['euclid_designation']
        expected_name = coord_to_name(np.array([row[ra_col]]),
                                      np.array([row[dec_col]]), epoch='J')[0]
        if expected_name == real_name:
            continue
        for survey, band in zip(surveys, bands):
            pattern = os.path.join(cutout_dir,
                                   f"{real_name}_{survey}_{band}*fov*.fits")
            for f in glob.glob(pattern):
                suffix = os.path.basename(f)[len(real_name):]
                new_path = os.path.join(cutout_dir, expected_name + suffix)
                if not os.path.exists(new_path):
                    os.symlink(os.path.abspath(f), new_path)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default=os.path.join(SCRIPT_DIR, 'configs',
                                             'Euclid_dr1_north.yaml'),
                        help='Path to the YAML config file with the inspector settings')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Relative paths inside the config (e.g. './data/...') are written
    # relative to this script's directory (examples/), not the current
    # working directory, so the tool runs the same regardless of where
    # it's invoked from.
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)

    my_candidate_df = pd.read_csv(resolve(cfg['df_path']), dtype={"oid": str})

    my_ra_column_name = cfg['ra']
    my_dec_column_name = cfg['dec']

    fov = cfg['fov']

    my_cutout_dir = cfg['cutout_path']

    surveys = cfg['surveys']
    bands = cfg['bands']

    visual_classes = cfg['visual_classes']

    verbosity = cfg['verbosity']

    saved_csv = resolve(cfg['saved_csv']) if cfg.get('saved_csv') else cfg.get('saved_csv')

    if cfg['euclid'] and 'euclid_designation' in my_candidate_df.columns:
        relink_to_expected_names(my_candidate_df, my_ra_column_name,
                                 my_dec_column_name, my_cutout_dir,
                                 surveys, bands)

    rgb_bands = cfg.get('rgb_bands')
    rgb_survey = cfg.get('rgb_survey')

    inspector_ts.run(my_candidate_df, my_cutout_dir, my_ra_column_name,
                  my_dec_column_name, surveys, bands,
                  # mag_column_names=mag_column_names,
                  # magerr_column_names=magerr_column_names,
                  # add_info_list=add_info_list,
                  minimum_fov=fov,
                  visual_classes=visual_classes, verbosity=verbosity,
                  euclid=cfg['euclid'], saved_csv=saved_csv,
                  rgb_bands=rgb_bands, rgb_survey=rgb_survey)
