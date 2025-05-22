# MVP_multimodal vae with mixture-of-prior

Official PyTorch implementation for MVP, introduced in the paper [Multimodal Disentanglement by Latent Variable Separation with Surrogate Modal Specifics and Mixture-of-Distributions Priors].


## Download datasets

For PolyMNIST and CUB Image-Captions dataset, please find download link to in [MMVAE+](https://github.com/epalu/mmvaeplus) repo.

For Tri-modal FashionMNIST dataset, download from [Google Drive](https://)

## Experiments
Run on PolyMNIST dataset
```
python ./polymnist/main_polymnist.py
```

Run on Tri-modal FashionMNIST dataset
```
python ./trimodal_fanshionmnist/main_polymnist.py
```


#### Acknowledgements
We thank the authors of the [MMVAE+](https://github.com/epalu/mmvaeplus) repo, from which our codebase is based, and from which we retrieve the link to the PolyMNIST and CUB Image-Captions dataset.
