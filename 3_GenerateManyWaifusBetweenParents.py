# generate many waifus between their parents.
# When TRUNCATION approaches 0, the appearance of the generated waifu is closer to the average of training samples.
# SEED1, SEED2 are integers between 0 and 2**32 - 1
# 0 < MORPH_FACTOR < 1

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

NUM_SPLIT = 39 # generate this number of waifus
TRUNCATION = 0.7
PARENT1 = [1224,5508,0.820513] # [SEED1, SEED2, MORPH_FACTOR]
PARENT2 = [1254,608,0] # [SEED1, SEED2, MORPH_FACTOR]

def randomLatents(seed1,seed2,morphingFactor,shape):
    # 0 < morphingFactor < 1, 0->latents0, 1-> latents1
    latents0 = np.random.RandomState(seed1).randn(1, shape)
    latents1 = np.random.RandomState(seed2).randn(1, shape)
    latents = latents0+(latents1-latents0)*morphingFactor
    return latents

def morphLatents(latents0, latents1, morphingFactor):
    latents = latents0+(latents1-latents0)*morphingFactor
    return latents

def createGif():
  from PIL import Image
  import glob
  files = sorted(glob.glob('results/*.png'))
  images = list(map(lambda file: Image.open(file), files))
  gif_filename = os.path.join(config.result_dir, 'stylegan.gif')
  images[0].save(gif_filename, save_all=True,
                 append_images=images[1:],
                 duration=2000/NUM_SPLIT, loop=0)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    _G, _D, Gs = pickle.load(open("./cache/2019-03-08-stylegan-animefaces-network-02051-021980.pkl", "rb"))

    # Print network details.
    Gs.print_layers()

    # Generate
    latents0 = randomLatents(int(PARENT1[0]), int(PARENT1[1]), PARENT1[2], Gs.input_shape[1])
    latents1 = randomLatents(int(PARENT2[0]), int(PARENT2[1]), PARENT2[2], Gs.input_shape[1])
    for i in range(NUM_SPLIT+1):
        latents = morphLatents(latents0, latents1, i/NUM_SPLIT)
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=TRUNCATION, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir,
                      'photo-{0}-{1}-{2:05f}-{3}-{4}-{5:05f}-{6:05f}'.format(*PARENT1,*PARENT2,i/NUM_SPLIT)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
    createGif()
