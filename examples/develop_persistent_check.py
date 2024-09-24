
import os
import numpy as np
from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.wcs import WCS

from IPython import embed

from wildhunt import image as whimg

import matplotlib.pyplot as plt

if __name__ == "__main__":

    folder = '~/Downloads'


    filename_0 = 'EUC_NIR_W-CAL-IMAGE_Y-1109-0_20240425T204019.768847Z.fits'
    filename_1 = 'EUC_NIR_W-CAL-IMAGE_Y-1109-1_20240425T204019.713622Z.fits'
    filename_2 = 'EUC_NIR_W-CAL-IMAGE_Y-1109-2_20240425T204019.729464Z.fits'
    filename_3 = 'EUC_NIR_W-CAL-IMAGE_Y-1109-3_20240425T204019.701395Z.fits'

    filenames = [filename_0, filename_1, filename_2, filename_3]
    cutouts = []
    ap_phot = []

    for filename in filenames:

        file_path = os.path.join(folder, filename)

        coord = SkyCoord(174.61753, 72.4415851, unit='deg', frame='icrs')


        # For each extension, open the fits file and extract the data
        hdul = fits.open(file_path)

        in_hduname = None
        for hdu in [h for h in hdul if 'SCI' in h.name]:
                # print(f"Extension {hdu.name}")
                header = hdu.header
                wcs = WCS(header)

                # Test if the coordinate is within the image
                x, y = wcs.world_to_pixel(coord)
                # print(f"Coordinate {coord} is at pixel {x}, {y}")
                if 0 < x < header['NAXIS1'] and 0 < y < header['NAXIS2']:
                    print(f"Coordinate {coord} is within the image")
                    print(f"Extension {hdu.name}")

                    in_hduname = hdu.name


                    img = whimg.Image(file_path, exten=in_hduname)
                    print(file_path)
                    cutout = img.get_cutout_image(coord.ra.value, coord.dec.value, 20)
                    zp_ab = img.header['ZPAB']
                    nanomag_correction = np.power(10, 0.4 * (22.5 - zp_ab))
                    cutouts.append(cutout)
                    phot = img.calculate_aperture_photometry(coord.ra.value, coord.dec.value,
                                                             nanomag_correction,
                                                             aperture_radii=np.array([1., 2.]),
                                                             exptime_norm=87.2248,
                                                             background_aperture=np.array([7, 10.]))
                    ap_phot.append(phot)

    fig, axes =  plt.subplots(2, 2, figsize=(10, 5))

    for idx, cutout in enumerate(cutouts):
        ax = axes.flatten()[idx]
        cutout._simple_plot(n_sigma=3, axis=ax, north=False)
        cutout._add_aperture_circle(ax, coord.ra.value, coord.dec.value, 2)
        # cutout._add_aperture_circle(ax, coord.ra.value, coord.dec.value, 7, edgecolor='blue')
        # cutout._add_aperture_circle(ax, coord.ra.value, coord.dec.value, 10, edgecolor='blue')
        string_ap_phot = '{:.4f}'.format(ap_phot[idx]['survey_band_flux_aper_2.0arcsec'])
        string_ap_phot_e = '{:.4f}'.format(ap_phot[idx]['survey_band_flux_err_aper_2.0arcsec'])
        ax.set_title(f"Aperture photometry:{string_ap_phot}+-{string_ap_phot_e}")

    plt.show()
    embed()




