import pandas as pd
from wildhunt import inspector_ts
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='./data/Euclid_northern_sources.csv')
    parser.add_argument('--ra', type=str, default='ra_J')
    parser.add_argument('--dec', type=str, default='dec_J')
    parser.add_argument('--fov', type=int, default=4)
    parser.add_argument('--cutout_path', type=str, default='/hs/babbage/data/group-schindler/ts/data/Euclid/cutout/sedm.mosaic_product/candidates/') # change cutout_path to the path you located
    parser.add_argument('--surveys', type=list, default=['Euclid', 'Euclid', 'Euclid', 'Euclid', 'Euclid'])
    parser.add_argument('--bands', type=list, default=['I', 'Z', 'Y', 'J', 'H'])
    parser.add_argument('--visual_classes', default={"VIS detection": ["no vis", "weak vis", "strong vis"], "Morphology": ["psf", "extended"], "Double source": ["only VIS", "VIS and NISP", "center + edge"], "Artifact": ["edge", "diffraction spike", "hot pixel", "masked", "other"]})
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=2)
    parser.add_argument('--euclid', type=bool, choices=[True, False], default=True)
    parser.add_argument('--saved_csv', type=str, default='Euclid_checked_candidates.csv')
    
    args = parser.parse_args()

    my_candidate_df = pd.read_csv(args.df_path, dtype={"oid": str})
    
    my_ra_column_name = args.ra
    my_dec_column_name = args.dec

    fov = args.fov

    my_cutout_dir = args.cutout_path

    surveys = args.surveys
    bands = args.bands

    visual_classes = args.visual_classes

    verbosity = args.verbosity

    inspector_ts.run(my_candidate_df, my_cutout_dir, my_ra_column_name,
                  my_dec_column_name, surveys, bands,
                  # mag_column_names=mag_column_names,
                  # magerr_column_names=magerr_column_names,
                  # add_info_list=add_info_list,
                  minimum_fov=fov,
                  visual_classes=visual_classes, verbosity=verbosity,
                  euclid=args.euclid, saved_csv=args.saved_csv)