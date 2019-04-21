# generate style mixing
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

TRUNCATION = 0.7
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# _Gs_cache = dict()

def load_Gs():
    _G, _D, Gs = pickle.load(open('./cache/2019-03-08-stylegan-animefaces-network-02051-021980.pkl' , "rb"))
    return Gs

def randomLatents(seed1,seed2,morphingFactor,shape):
    # 0 < morphingFactor < 1, 0->latents0, 1-> latents1
    latents0 = np.random.RandomState(seed1).randn(shape)
    latents1 = np.random.RandomState(seed2).randn(shape)
    latents = latents0+(latents1-latents0)*morphingFactor
    return latents

def draw_style_mixing_figure(png_filename, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png_filename)
    src_latents = np.stack(randomLatents(seed[0],seed[1],seed[2],Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(randomLatents(seed[0],seed[1],seed[2],Gs.input_shape[1]) for seed in dst_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, truncation_psi=TRUNCATION, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, truncation_psi=TRUNCATION, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * 5), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png_filename)

# --------------- main -----------------
def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    draw_style_mixing_figure(os.path.join(config.result_dir, 'style-mixing.png'), load_Gs(),
                         w=512, h=512, style_ranges=[range(0,4)]*1+[range(4,8)]*1+[range(8,16)]*1,
                         src_seeds=[
                             [100,101,0.3],
                             [720,104,0.4],
                             [587,22,0.6]
                         ],
                         dst_seeds=[
                             [78,30,0.1],
                             [6872,3,0.6],
                             [729,9,0.9]
                         ])
    # src:columns, dst:rows.
    # style_ranges must match number of rows

if __name__ == "__main__":
    main()
