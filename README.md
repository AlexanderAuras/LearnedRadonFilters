# Convergent Data-driven Regularizations for CT Reconstruction
<font size="3">Samira Kabri<sup>1</sup>, Alexander Auras<sup>2</sup>, Danilo Riccio<sup>3</sup>, Martin Benning<sup>3,4</sup>, Michael Moeller<sup>2</sup> and Martin Burger<sup>1</sup></font>

<font size="2">
<sup>1</sup>Department of Mathematics, University of Erlangen-Nuremberg<br/>
<sup>2</sup>Institute for Vision and Graphics, University of Siegen<br/>
<sup>3</sup>School of Mathematical Sciences, Queen Mary University of London<br/>
<sup>4</sup>The Alan Turing Institute, British Library<br/>
</font><br/>

This repository contains the official pytorch implementation of *Convergent Data-driven Regularizations for CT Reconstruction*.

<!--[![arXiv](http://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000)-->
![Python](http://img.shields.io/badge/python-%3E%3D3.8-blue)
![PyTorch](http://img.shields.io/badge/PyTorch-%3E%3D1.12-blue)

## Features
- Calculation of $\Pi$, $\Delta$ and $\Gamma$
- Calculation of the analytic coefficients $g_n$
- Learning of the coefficients $\overline{g_n}$
- Calculation of the analytic filter $\rho$
- Learning of the filter $\overline{\rho}$
- Logging of results via tensorboard
- Saving of results in *.pt* files

## Dependencies
Dependencies to execute the calculations and the used versions:
- torch (1.12.1)
- hydra-core (1.2.0)
- pytorch-lightning (1.7.1)
- torchmetrics (0.9.3)
- radon (1.0.0) (either as submodule or from the repository at [https://github.com/AlexanderAuras/radon](https://github.com/AlexanderAuras/radon))

Optional dependencies for the visualizations: 
- notebook
- matplotlib
- tensorflow

Dependencies to view tensorboard results:
- tensorboard 

## Installation
- Install conda and create environment (optional)
- Install dependencies
- Clone repository: `git clone https://github.com/AlexanderAuras/LearnedRadonFilters`
- Adjust the output directory in *configs/default.yaml* and the paths in *mnist_datamodule.py*

## Usage
- Adjust the configurations in the *configs* folder
    - Adjust configurations for datasets in *configs/dataset/\<dataset\>.yaml*
    - Adjust configurations for the models in *configs/model/\<model\>.yaml*
    - Adjust general configurations in *configs/default.yaml*
- Execute the training/calculations with `python main.py hydra.job.name=<name>`
- View tensorboard results via `tensorboard --logdir=<output_directory>`
- View saved *.pt* files in the jupyter notebooks *plots.ipynb*, *test.ipynb* or *visualization.ipynb*
    - For every run $\Pi$, $\Delta$ and $\Gamma$ are saved in *pi.pt*, *delta.pt* and *gamma.pt* 
    - For FFT-runs, *coefficients.pt* contains $\rho$ or $\overline{\rho}$
    - For SVD-runs, *coefficients.pt* contains $g_n$ or $\overline{g_n}$. 
    - For SVD-runs the singular values are saved in *singular_values.pt* 