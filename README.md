# [DISGAN: Wavelet-informed discriminator guides GAN to MRI Super-resolution with noise cleaning ](https://arxiv.org/abs/2308.12084)

@ICCV2023

---
Requirements:
* multiprocessing
* torch2
* pytorch-lightning

## Train from scratch
~~~python
python ln_DDP_train.py --model_name 'DWT_D'
~~~

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
