import os
import tempfile
from pathlib import Path
import cog
import numpy as np
import dnnlib
import PIL.Image
import torch
import math
import legacy
import glob
import cv2
import random
import shutil


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda')
        self.models = {}
        network_pkls = ['afhqcat5k256x256-apa.pkl',
                        'anime5k256x256-apa.pkl',
                        'cub12k256x256-apa.pkl',
                        'ffhq5k1024x1024-apa.pkl',
                        'ffhq5k256x256-apa.pkl',
                        'ffhq70kfull256x256-apa.pkl']
        for network_pkl in network_pkls:
            with dnnlib.util.open_url(f'models/{network_pkl}') as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
                self.models[network_pkl.split('.')[0]] = G

    @cog.input(
        "model",
        type=str,
        default='ffhq70kfull256x256-apa',
        options=['afhqcat5k256x256-apa', 'anime5k256x256-apa', 'cub12k256x256-apa', 'ffhq5k1024x1024-apa',
                 'ffhq5k256x256-apa', 'ffhq70kfull256x256-apa'],
        help="choose model for image generation",
    )
    @cog.input(
        "num_images",
        type=int,
        default=4,
        options=[1, 4, 9, 16],
        help="number of images to generate, valid when seed is not provided",
    )
    @cog.input(
        "truncation_psi",
        type=float,
        default=1,
        help="Truncation psi",
    )
    @cog.input(
        "noise_mode",
        type=str,
        options=['const', 'random', 'none'],
        default='const',
        help="Noise mode",
    )
    def predict(self, model, num_images, truncation_psi, noise_mode):
        seeds = random.sample(range(1, 1001), num_images)
        G = self.models[model]

        # Labels.
        label = torch.zeros([1, G.c_dim], device=self.device)
        class_idx = None
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')
        output_dir = 'output'
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Generate images.
            for seed_idx, seed in enumerate(seeds):
                print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(self.device)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output_dir}/seed{seed:04d}.png')
            img_list = sorted(glob.glob(os.path.join(output_dir, '*')))

            out_path = Path(tempfile.mkdtemp()) / "out.png"
            imgs = [cv2.imread(img) for img in img_list]
            if model == 'ffhq5k1024x1024-apa':
                im_v = cv2.vconcat(imgs)
                cv2.imwrite(str(out_path), im_v)
            else:
                image_grid = save_image_grid(int(math.sqrt(num_images)), imgs)
                cv2.imwrite(str(out_path), image_grid)
        finally:
            clean_folder(output_dir)
        return out_path


def save_image_grid(dim, images):
    assert len(images) == dim * dim, 'the number of images does not fit the grid dimensions'
    image_grid = []
    row = 0
    for i in range(dim):
        image_grid.append(images[row * dim: (row + 1) * dim])
        row += 1
    final_image = cv2.vconcat([cv2.hconcat(imgs_h) for imgs_h in image_grid])
    return final_image


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
