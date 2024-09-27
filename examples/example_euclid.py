#!/usr/bin/env python

import time

from wildhunt import catalog, pypmsgs

msgs = pypmsgs.Messages()


# download Euclid images
def example_download_images():
    t0 = time.time()

    cat = catalog.Catalog(
        "example",
        "RA",
        "DEC",
        "Name",
        datapath="/Users/francesco/repo/WildHunt/examples/data/Euclid_sources.csv",
    )

    survey_dict = [
        {"survey": "Euclid", "bands": ["VIS", "Y", "J", "H"], "fov": 10},
    ]

    cat.get_survey_images(
        "/Users/francesco/data/wildhunt/example_cutouts_euclid/", survey_dict, n_jobs=3
    )
    msgs.info(f"Took {time.time() - t0:.1f}s to download the requested cutouts.")


if __name__ == "__main__":
    example_download_images()
