#To start you have to import the library pyvo (it is also possible to use astroquery if you want)
import pyvo

## To perform a TAP query you have to connect to the service first
tap_service = pyvo.dal.TAPService('https://vo.astron.nl/__system__/tap/run/tap')

# This works also for
from pyvo.registry.regtap import ivoid2service
vo_tap_service = ivoid2service('ivo://astron.nl/tap')[0]

# The TAPService object provides some introspection that allow you to check the various tables and their
# description for example to print the available tables you can execute
print('Tables present on http://vo.astron.nl')
for table in tap_service.tables:
   print(table.name)
print('-' * 10 + '\n' * 3)

# or get the column names
print('Available columns in lotss_dr2.mosaics')
print(tap_service.tables['lotss_dr2.mosaics'].columns)
print('-' * 10 + '\n' * 3)

## You can obviously perform tap queries accross the whole tap service as an example a cone search
print('Performing TAP query')
result = tap_service.search(
   "SELECT TOP 5 source_name, beam_number, accref, centeralpha, centerdelta, "
   "obsid, DISTANCE(" \
       "POINT('ICRS', centeralpha, centerdelta),"\
       "POINT('ICRS', 208.36, 52.36)) AS dist"\
   " FROM lotss_dr2.main_sources"  \
   " WHERE 1=CONTAINS("
   "    POINT('ICRS', centeralpha, centerdelta),"\
   "    CIRCLE('ICRS', 208.36, 52.36, 0.08333333)) "\
   " ORDER BY dist ASC"
   )
print(result)

# The result can also be obtained as an astropy table
astropy_table = result.to_table()
print('-' * 10 + '\n' * 3)

## You can also download and plot the image
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import requests, os
import numpy as np

# DOWNLOAD only the first result
#
print('Downloading only the first result')
print(result[0]['obsid'])
file_name = '{}_{}_{}.fits'.format(
    result[0]['obsid'],
    result[0]['target'],
    result[0]['beam_number'])
path = os.path.join(os.getcwd(), file_name)
http_result = requests.get(result[0]['accref'])
print('Downloading file in', path)
with open(file_name, 'wb') as fout:
   for content in http_result.iter_content():
       fout.write(content)
hdu = fits.open(file_name)[0]
wcs = WCS(hdu.header)
# dropping unnecessary axes
wcs = wcs.dropaxis(2).dropaxis(2)
plt.subplot(projection=wcs)
plt.imshow(hdu.data[0, 0, :, :], vmax=0.0005)
plt.xlabel('RA')
plt.ylabel('DEC')
plt.show()