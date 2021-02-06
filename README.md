##  Efficient Approximate Inference for Stationary Kernel on Frequency Domain

We provide the implementation and experiment results code for the paper Efficient Approximate Inference for Stationary Kernel on Frequency Domain.


## Description

* main_exp1 in section 5-1.ipynb : kernel approximation experiemnt to validate sampling strategy (expeiriment section 5.1)
* main_exp2-1 in section 5-2.ipynb, main_exp2-2 in section 5-2.ipynb : kernel approximation in training (expeiriment section 5.2)
* main_ablationstudy_exp.py. run_ablation_exp.sh : ablation study (experiment section 5.3)
* main_uci_regression.py, main_uci_regression_batch.py : a large-scale and hight-dimensional uci regression (experiment section 5.4)



## Requirements

* python >= 3.6
* torch >= 1.7
* pandas
* scipy


## Dataset

* [UCI Wilson dataset][https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view]



## Installation

    git clone https://github.com/ABInferGSM/src.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 




## Reference 

* https://github.com/idelbrid/Randomly-Projected-Additive-GPs
* https://github.com/yaringal/VSSGP
* https://github.com/GPflow/GPflow/blob/develop/gpflow/models/svgp.py





