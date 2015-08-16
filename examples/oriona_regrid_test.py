from __future__ import print_function

from astropy.io import fits
from astropy import convolution
from scipy.ndimage import zoom
import numpy as np

from fil_finder import fil_finder_2D

'''
Regrid OrionA-South to distance of Lupus
'''

img, hdr = fits.getdata("orionA-C-350.fits", header=True)
img = img + 35.219

distance = 400.
beamwidth = 24.9

orig_img = img.copy()
orig_distance = 400.
orig_beamwidth = 24.9
orig_hdr = hdr.copy()

# Convolve to the distance of IC-5146 (460 pc)
convolve_to_common = False
regrid_to_common = True
if convolve_to_common:
    r = 460. / float(distance)
    if r != 1.:
        conv = np.sqrt(r ** 2. - 1) * \
            (beamwidth / np.sqrt(8*np.log(2)) / (np.abs(hdr["CDELT2"]) * 3600.))
        if conv > 1.5:
            kernel = convolution.Gaussian2DKernel(conv)
            good_pixels = np.isfinite(img)
            nan_pix = np.ones(img.shape)
            nan_pix[good_pixels == 0] = np.NaN
            img = convolution.convolve(img, kernel, boundary='fill',
                                       fill_value=np.NaN)
            # Avoid edge effects from smoothing
            img = img * nan_pix

            beamwidth *= conv

        else:
            print("No convolution support.")

if regrid_to_common:

    r = float(distance) / 140.

    if r != 1:

        good_pixels = np.isfinite(img)
        good_pixels = zoom(good_pixels, round(r, 3),
                           order=0)

        img[np.isnan(img)] = 0.0
        regrid_conv_img = zoom(img, round(r, 3))


        nan_pix = np.ones(regrid_conv_img.shape)
        nan_pix[good_pixels == 0] = np.NaN


        img = regrid_conv_img * nan_pix

        distance = 140.

        hdr['CDELT2'] /= r

filfind = fil_finder_2D(img, hdr, beamwidth,
                        distance=distance, glob_thresh=20)

print(filfind.beamwidth, filfind.imgscale)

filfind.create_mask(test_mode=False)
filfind.medskel(verbose=False)

# filfind.analyze_skeletons()
# filfind.find_widths(verbose=True)

# Now run the original, zoom the mask, and compare skeletons

filfind_orig = fil_finder_2D(orig_img, orig_hdr, orig_beamwidth,
                             distance=orig_distance, glob_thresh=20)

filfind_orig.create_mask()

filfind_orig.mask = zoom(filfind_orig.mask[10:-10, 10:-10], round(r, 3), order=0)

from fil_finder.utilities import padwithzeros
filfind_orig.mask = np.pad(filfind_orig.mask, 10, padwithzeros)
from scipy.ndimage import median_filter, label
filfind_orig.mask = median_filter(filfind_orig.mask, 5)

filfind_orig.medskel()


print("Regrid " + str(label(filfind.skeleton, np.ones((3, 3)))[1]))
print("Original " + str(label(filfind_orig.skeleton, np.ones((3, 3)))[1]))

import matplotlib.pyplot as p

p.imshow(filfind.flat_img, interpolation='nearest', origin='lower', cmap='gray')
p.contour(filfind.skeleton, colors='r', alpha=0.7, linewidths=3)
p.contour(filfind_orig.skeleton, colors='b', alpha=0.7)

p.show()

