

from wildhunt import image as  whim
import matplotlib.pyplot as plt

if __name__ == '__main__':

    filename = 'J080008.92+103115.65_UHS_J_fov150.fits'

    # Instantiate the image class
    img = whim.Image(filename, exten=1)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    img._simple_plot(axis=ax, north=False)
    plt.show()