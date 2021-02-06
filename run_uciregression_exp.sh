#!/bin/bash 


numcuda=3
setdata=gas
setQ=4
ratesamplespt=.01
lrhyp=.01
iter=2000
numrepexp=5
datanormal=True

echo 'run exp'
{
CUDA_VISIBLE_DEVICES=$numcuda python3 main_uci_regression.py --filename $setdata --numQ $setQ --numspt 15 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal ;
CUDA_VISIBLE_DEVICES=$numcuda python3 main_uci_regression.py --filename $setdata --numQ $setQ --numspt 10 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal ;
CUDA_VISIBLE_DEVICES=$numcuda python3 main_uci_regression.py --filename $setdata --numQ $setQ --numspt 3 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal ;
CUDA_VISIBLE_DEVICES=$numcuda python3 main_uci_regression.py --filename $setdata --numQ $setQ --numspt 5 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal ;
CUDA_VISIBLE_DEVICES=$numcuda python3 main_uci_regression.py --filename $setdata --numQ $setQ --numspt 7 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal ;









