# Time-multiplexed Neural Holography: A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators <br> SIGGRAPH 2022
### [Project Page](http://www.computationalimaging.org/publications/time-multiplexed-neural-holography/) | [Video](https://youtu.be/k2dg-Ckhk5Q) | [Paper](https://drive.google.com/file/d/1n8xSdHgW0D5G5HhwSKrqCy1iztAcDHgX/view?usp=sharing)
PyTorch implementation of <br>
[Time-multiplexed Neural Holography: A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators](http://www.computationalimaging.org/publications/time-multiplexed-neural-holography/)<br>
 [Suyeon Choi](http://stanford.edu/~suyeon/)\*,
 [Manu Gopakumar](https://www.linkedin.com/in/manu-gopakumar-25032412b/)\*,
 [Yifan Peng](http://web.stanford.edu/~evanpeng/),
 [Jonghyun Kim](http://j-kim.kr/),
 [Matthew O'Toole](https://www.cs.cmu.edu/~motoole2/),
 [Gordon Wetzstein](https://computationalimaging.org)<br>
  \*denotes equal contribution  
in SIGGRAPH 2022

<img src='img/teaser.png'/>

## Get started
Our code uses [PyTorch Lightning](https://www.pytorchlightning.ai/) and PyTorch >=1.10.0.

You can set up a conda environment with all dependencies like so:
```
conda env create -f env.yml
conda activate tmnh
```

## High-Level structure
The code is organized as follows:


`./`
* ```main.py``` generates phase patterns from LF/RGBD/RGB data using SGD.
* ```holo2lf.py``` contains the Light-field ↔ Hologram conversion implementations.
* ```algorithms.py``` contains the gradient-descent based algorithm for LF/RGBD/RGB supervision

* ```params.py``` contains our default parameter settings. :heavy_exclamation_mark:**(Replace values here with those in your setup.)**:heavy_exclamation_mark:

* ```quantization.py``` contains modules for quantizations (projected gradient, sigmoid, Gumbel-Softmax).
* ```image_loader.py``` contains data loader modules.
* ```utils.py``` has some other utilities.




`./props/` contain the wave propagation operators (in simulation and physics).

`./hw/` contains modules for hardware control and homography calibration
* ```ti.py``` contains data given by Texas Instruments.
* ```ti_encodings.py``` contains phase encoding and decoding functionalities for the TI SLM.


## Run
To run, download the sample images from [here](https://drive.google.com/file/d/1aooTbzsmGw-Rfel7ntb1HJY1kILLSuEk/view?usp=sharing) and place the contents in the `data/` folder.

### Dataset generation / Model training
Please see the [supplement](https://drive.google.com/file/d/1n9hdLq1xvur4I_OkGNyFgoKHGDZcMxcE/view) and [Neural 3D Holography repo](https://github.com/computational-imaging/neural-3d-holography) for more details on dataset generation and model training.
```
# Train TMNH models
for c in 0 1 2
do
  python train.py -c=configs_model_training.txt --channel=$c --data_path=${dataset_path}
done

```


### Run SGD with various target distributions (RGB images, focal stacks, and light fields)
```
for c in 0 1 2
do
  # 2D rgb images
  python main.py -c=configs_2d.txt --channel=$c
  # 3D focal stacks
  python main.py -c=configs_3d.txt --channel=$c
  # 4D light fields
  python main.py -c=configs_4d.txt --channel=$c
done
```

### Run SGD with advanced quantizations
```
q=gumbel-softmax; # try none, nn, nn_sigmoid as well.
python main.py -c=configs_2d.txt --channel=$c --quan_method=$q

```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{choi2022time,
               author = {Choi, Suyeon
                         and Gopakumar, Manu
                         and Peng, Yifan
                         and Kim, Jonghyun
                         and O'Toole, Matthew
                         and Wetzstein, Gordon},
               title={Time-multiplexed neural holography: a flexible framework for holographic near-eye displays with fast heavily-quantized spatial light modulators},
               booktitle={ACM SIGGRAPH 2022 Conference Proceedings},
               pages={1--9},
               year={2022}
}
```

## Acknowledgmenets
Thanks to [Brian Chao](https://bchao1.github.io/) for the help with code updates and [Cindy Nguyen](https://ccnguyen.github.io) for helpful discussions. This project was in part supported by a Kwanjeong Scholarship, a Stanford SGF, Intel, NSF (award 1839974), a PECASE by the ARO (W911NF-19-1-0120), and Sony.

## Contact
If you have any questions, please feel free to email the authors.