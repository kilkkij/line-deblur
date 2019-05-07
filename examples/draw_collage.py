
import argparse
import matplotlib.pyplot as plt

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
if args.latent is not None:
	images.append(imtools.imread(args.latent))
	names.append('latent')
if args.kernel is not None:
	images.append(imtools.imread(args.kernel))
	names.append('est. kernel')

fig, axes = plt.subplots(1, len(images), figsize=(3*len(images)+1, 3.5))
for ax, name, image in zip(axes, names, images):
	imtools.plot_image(image, name, ax)

fig.tight_layout()
if args.out_file_name is not None:
	fig.savefig(args.out_file_name, dpi=450)
	print(args.out_file_name)
else:
	plt.show()