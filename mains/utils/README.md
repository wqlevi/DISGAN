# Util module: a module for auxilary function

* multiprocessing for preprocessing
* metrics calculation and recording
* adding noise to simulate noisy images for denoising

# Inference

1. Generate LR whole brain volume
2. Infer them to get whole brain SR volume
3. Suggested to be used on `device=cpu` when your GPU mem is low

or

1. Take a whole brain volume, downsample to LR
2. Patch them into overlapped volumes
3. Infer individual LR patches to get SR patches
4. Assemble them to get a SR whole brain volume
