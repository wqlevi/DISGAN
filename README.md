# [DISGAN: Wavelet-informed discriminator guides GAN to MRI Super-resolution with noise cleaning ](https://arxiv.org/abs/2308.12084)

@ICCV2023

- [Environment](#Environment)
- [Super-resolution](#Super-resolution)
- [Denoising](#denoising)
- [Train](#train-from-scratch)
- [Cite](#citation)
---
### Environment
Essentially required packaged:
* multiprocessing
* torch2
* pytorch-lightning

### Super-resolution

<figure id="ours" class=subfigure style="display: inline-block; width=30%">
<img src="./imgs/Ours.png" alt="DISGAN" width="200"/>
<figcaption class=subcaption, style="text-align: center"><strong>DISGAN(ours)</strong></figcaption>
</figure>
<figure id="gt" class=subfigure style="display: inline-block; width=30%">
<img src="./imgs/GT.png" alt="GT" width="200"/>
<figcaption class=subcaption, style="text-align: center">GT</figcaption>
</figure>


### Denoising
<figure id="sim" class=subfigure style="display: inline-block; width=30%">
<img src="./imgs/sim_noise_whole.png" alt="DISGAN" width="600"/>
<figcaption class=subcaption><strong>Top:</strong>DISGAN(ours) for denoising; <strong>bottom:</strong>GT with noise</figcaption>
</figure>

### Train from scratch
~~~bash
python ln_DDP_train.py --model_name 'DWT_D'
~~~

### Citation
~~~bibtex
@misc{wang2023disgan,
      title={DISGAN: Wavelet-informed Discriminator Guides GAN to MRI Super-resolution with Noise Cleaning}, 
      author={Qi Wang and Lucas Mahler and Julius Steiglechner and Florian Birk and Klaus Scheffler and Gabriele Lohmann},
      year={2023},
      eprint={2308.12084},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
~~~
