
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../core')
import imtools

parser = argparse.ArgumentParser()
parser.add_argument('obs', type=str, help='observation')
parser.add_argument('deblurred', type=str, help='deblurred output')
parser.add_argument('-l', dest='latent', type=str, help='latent image')
parser.add_argument('-k', dest='kernel', type=str, help='estimated kernel')
parser.add_argument('-o', dest='out_file_name', type=str, help='output file name')
args = parser.parse_args()

images = [
    imtools.imread(args.obs),
    imtools.imread(args.deblurred)
    ]
names = [
    'obs',
    'estimate'
    ]
widths = [1, 1]
if args.latent is not None:
    images.append(imtools.imread(args.latent))
    names.append('latent')
    widths.append(1)
if args.kernel is not None:
    raw_image = imtools.imread(args.kernel)
    images.append(raw_image/np.max(raw_image))
    names.append('est. kernel')
    widths.append(.5)

gs = gridspec.GridSpec(1, len(images), width_ratios=widths)
fig = plt.figure(figsize=(3*len(images)+1, 3.5))
axes = [plt.subplot(gsi) for gsi in gs]

for ax, name, image in zip(axes, names, images):
    imtools.plot_image(image, name, ax)

fig.tight_layout()
if args.out_file_name is not None:
    fig.savefig(args.out_file_name, dpi=450)
    print(args.out_file_name)
else:
    plt.show()