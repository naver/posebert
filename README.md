# Leveraging MoCap Data for Human Mesh Recovery [3DV 2021]

[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2110.09243)

> [**Leveraging MoCap Data for Human Mesh Recovery**](https://arxiv.org/abs/2110.09243),            
> [Fabien Baradel*](https://fabienbaradel.github.io/),
> [Thibaut Groueix*](https://imagine.enpc.fr/~groueixt/),
> [Philippe Weinzaepfel](https://europe.naverlabs.com/people_user/philippe-weinzaepfel/),
> [Romain Brégier](https://europe.naverlabs.com/people_user/romain-bregier/),
> [Yannis Kaltandidis](https://www.skamalas.com/),
[Grégory Rogez](https://europe.naverlabs.com/people_user/gregory-rogez/)       
> *International Conference on 3D Vision (3DV), 2021*

Pytorch demo code and pre-trained models for MoCap-SPIN and PoseBERT.

## Install

Our code is running using python3.7 and requires the following packages:

- pytorch-1.7.1+cu110
- pytorch3d-0.3.0
- torchvision
- opencv
- PIL
- numpy
- smplx
- einops
- roma

We do not provide support for installation.

## Download models

First download our models by running the following command:

```
# MocapSpin & PoseBERTs
wget http://download.europe.naverlabs.com/leveraging_mocap_models/models.tar.gz
tar -xvf models.tar.gz
rm models.tar.gz

# DOPE real time
wget http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz
mv DOPErealtime_v1_0_0.pth.tgz models/
```

This will create a folder ```models``` which should contains the following files:
- ```mocapSPIN.pt```: an image-based model for estimating SMPL parameters.
- ```posebert_smpl.pt```: a video-based model which is smoothing SMPL parameters estimated from a image-based model.
- ```posebert_h36m.pt```: a video-based model which is estimating SMPL parameters estimated from a sequence of 3d poses
  in H36M format.

You also need to download a regressor and mean parameters, please download them using the following links and place them into the ```models``` directory:
- [J_regressor_h36m.npy](https://drive.google.com/file/d/1ZPVtzWRcN97U4c8F4ZOm9tAzibQQxlqS/view?usp=sharing)
- [smpl_mean_params.npz](https://download.openmmlab.com/mmpose/datasets/smpl_mean_params.npz)

Finally you need to download by yourself ```SMPL_NEUTRAL.pkl``` from
the [SMPLify website](https://smplify.is.tue.mpg.de/index.html) and place it into ```models```.

The ```models``` directory tree should looks like this:
```
models
├── SMPL_NEUTRAL.pkl
├── J_regressor_h36m.npy
├── smpl_mean_params.npz
├── mocapSPIN.pt
├── posebert_smpl.pt
├── posebert_h36m.pt
├── DOPErealtime_v1_0_0.pth.tgz
```

## Demo

We provide a demo code which is recovering offline the human mesh from a RGB video. To use our code on a video, use the
following command:

```
python demo.py --method <methodname> --video <videoname> --sideview <sideview-or-not>
```

with

- ```<methodname>```: name of model to use (```mocapspin_posebert```, ```mocapspin``` or ```dope_posebert```)
- ```<videoname>```: location of the video to test
- ```<sideview>```: if you want to render the sideview (0 or 1)

The command will create a video ```<videoname>_<methodname>.mp4``` which shows the estimated human mesh.

## Disclaimer

We do not handle multi-person human mesh recovery and we do not use a tracking algorithm. Thus for each timestep we take
into account only the first person detected in the scene using DOPE.

## Citation

If you find our work useful please cite our paper:

```
@inproceedings{leveraging_mocap,
  title={Leveraging MoCap Data for Human Mesh Recovery},
  author={Baradel*, Fabien and Groueix*, Thibault and Weinzaepfel, Philippe and Br\'egier and Kalantidis, Yannis and Rogez, Gr\'egory},
  booktitle={3DV},
  year={2021}
}
```

## License

MoCapSPIN and PoseBERT are distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
