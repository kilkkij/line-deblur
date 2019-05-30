python3 generate_test_case.py original-photos/library.jpg library-nonoise
python3 ../core/deconvolve.py cases/library-nonoise/obs.png cases/library-nonoise/deblurred.png -k cases/library-nonoise/kernel.png
python3 draw_collage.py cases/library-nonoise/obs.png cases/library-nonoise/deblurred.png -l cases/library-nonoise/latent.png -k cases/library-nonoise/kernel.png -o cases/library-nonoise/collage.png

python3 generate_test_case.py original-photos/library.jpg library --noise 0.010
python3 ../core/deconvolve.py cases/library/obs.png cases/library/deblurred.png -k cases/library/kernel.png
python3 draw_collage.py cases/library/obs.png cases/library/deblurred.png -l cases/library/latent.png -k cases/library/kernel.png -o cases/library/collage.png

python3 generate_test_case.py original-photos/lena.jpg lena --noise 0.010
python3 ../core/deconvolve.py cases/lena/obs.png cases/lena/deblurred.png -k cases/lena/kernel.png
python3 draw_collage.py cases/lena/obs.png cases/lena/deblurred.png -l cases/lena/latent.png -k cases/lena/kernel.png -o cases/lena/collage.png

python3 ../core/deconvolve.py original-photos/bike.jpg cases/bike/deblurred.png -k cases/bike/kernel.png
python3 draw_collage.py original-photos/bike.jpg cases/bike/deblurred.png -k cases/bike/kernel.png -o cases/bike/collage.png