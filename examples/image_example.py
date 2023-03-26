

from wildhunt import image as  whim
import matplotlib.pyplot as plt

if __name__ == '__main__':

    filename = 'J080008.92+103115.65_UHS_J_fov150.fits'

    # Instantiate the image class
    img = whim.Image(filename, exten=1)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    img._simple_plot(axis=ax, north=False)
    img._add_aperture_circle(ax, img.ra, img.dec, 10, edgecolor='red')
    img.add_aperture_rectangle(ax, img.ra, img.dec, 10, 20,
                               angle=20, edgecolor='blue')
    img._add_slit(ax, img.ra, img.dec, 12, 30,
                               angle=20, edgecolor='green')
    plt.show()