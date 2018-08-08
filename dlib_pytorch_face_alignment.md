# dlib + PyTorch + face-alignment

## Troubleshooting

#### Segmentation fault error
* Cause:
* Solution: Compile `dlib` with `gcc>=4.9`. With `conda`, you can update `gcc` as follows:
```bash
conda uninstall gcc
conda install -c conda-forge isl=0.17.1
anaconda search -t conda gcc-5
conda install -c bonsai-team gcc-5
```
* Reference: [face-alignment github](https://github.com/1adrianb/face-alignment/issues/85)
