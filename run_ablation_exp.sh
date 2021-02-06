#!/bin/bash 



numcuda=0
setdata=skillcraft  #parkinsons,pol

setQ=3
ratesamplespt=.05
#lrhyp=.005
lrhyp=.005
iter=3000
numrepexp=5
datanormal=True


echo 'run exp'
{
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ --numspt 3 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ --numspt 7 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ --numspt 15 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
}




setQ2=6

echo 'run exp'
{
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ2 --numspt 3 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ2 --numspt 7 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
CUDA_VISIBLE_DEVICES=$numcuda python3 main_ablationstudy_exp.py --filename $setdata --numQ $setQ2 --numspt 15 --numbatch 1 --ratesamplespt $ratesamplespt --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp --datanormal $datanormal &
}




