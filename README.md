##  Efficient Approximate Inference for Stationary Kernel on Frequency Domain

We provide the implementation and experiment results for the paper Efficient Approximate Inference for Stationary Kernel on Frequency Domain.


## Description

### Methods

* models/gp_rrff_reg.py : a sampling based variational inference (SVSS)
* models_utility/spt_manager_train.py : a sampling strategy of Propoistion 2 
* models_utility/personalized_adam.py : an approximate natural gradient update of Proposition 3

### Experiments

* main_exp1 in section 5-1.ipynb : kernel approximation experiemnt to validate sampling strategy (expeiriment section 5.1)
* main_exp2-1 in section 5-2.ipynb, main_exp2-2 in section 5-2.ipynb : kernel approximation in training (experiment section 5.2)
* main_ablationstudy_exp.py : ablation study (experiment section 5.3)
* main_uci_regression.py, main_uci_regression_batch.py : a large-scale and high-dimensional UCI dataset regression (experiment section 5.4)
* supplementary SM kernel Recovery by SVSS.ipynb : SM kernel Recovery conducted in [experiment section 5.1](https://arxiv.org/pdf/1910.13565.pdf)


## Requirements

* python >= 3.6
* torch = 1.7
* pandas
* scipy


## Dataset

* datasets/uci_datasets/ : kin8nm and parkinsons set used in our experiment
* datasets/uci_wilson/ : download [UCI Wilson dataset](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view) and unzip the downloaded file


## Installation

    git clone https://github.com/ABInferGSM/src.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 

* https://github.com/yaringal/VSSGP
* http://www.tsc.uc3m.es/~miguel/downloads.php 
* https://github.com/GPflow/GPflow/blob/develop/gpflow/models/svgp.py
* https://github.com/GAMES-UChile/mogptk/
* https://github.com/idelbrid/Randomly-Projected-Additive-GPs




