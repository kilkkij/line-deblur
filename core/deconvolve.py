
import argparse

import imtools
import optimization

def main(raw_args=None):

    parser = argparse.ArgumentParser(description='Deconvolve image.')
    parser.add_argument('input', type=str, help='input file path')
    parser.add_argument('output', type=str, default='deblurred-image.jpg', help='output file path')
    parser.add_argument('-k', dest='kernel_path', type=str, default=None, help='output file path')
    args = parser.parse_args(raw_args)
    
    obs_data = imtools.imread(args.input)
    latent_estimate, kernel_estimate = optimization.optimize(obs_data)

    imtools.imsave(latent_estimate, args.output)
    if args.kernel_path is not None:
        imtools.imsave(kernel_estimate, args.kernel_path)

if __name__ == '__main__':
    main()