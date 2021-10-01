# SNH
This repository contains the implementation of Spatial Neural Histograms [1] to answer range count queries on a geospatial dataset while preserving differential privacy. Neural networks are trained using JAX and are used to answer RCQs.

## Instalation and requirements

## Instalation and requirements
### Install conda environment and python targets:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

### create a python target and source into python enivorment
conda create -n [name_of_enviornment] python=3.8
conda activate [name_of_enviornment]

### install jax lib 0.1.71 with CUDA support (without GPU is also okay, ~10x performance penalty with 15 core CPU) 
wget https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.71+cuda111-cp38-none-manylinux2010_x86_64.whl
pip install jaxlib-0.1.71+cuda111-cp38-none-manylinux2010_x86_64.whl

### install other libraries
pip install jax==0.2.12 numpy pandas dm-haiku sklearn rtree

## Running SNH
Running SNH can be done by calling python run.py. SNH configureations are set in run.py, through the python dictionary config. Specifically, the dictionary contains a key 'NAME'. When calling python run.py, the code creates the folder tests/config['NAME'], where the result of the experiment is written. Explanation of each of the configurations is available in file run.py.

## SNH output
SNH trains config['no_models'] number of different models. The i-th model's training and testing statistic is written in the file tests/config['NAME']/i/out.txt

## Example
The folder data contains two datasets: CABS_SFS.npy (from [2]) and gowalla_SF.npy (from [3]). We consider releasing CABS_SFS.npy with differential privacy while using gowalla_SF.npy as an auxiliary public dataset. Calling python run.py performs training and testing with this setting. For example, the result of the zero-th trained model will be written in tests/test_sf_cabs/0/out.txt. A sample output for that file is 

>Creating model for query size 0.2375±0.0375 % of query space
>
>Preparing training data
>
>Calculating training weights
>
>initializing the model
>
>training
>100 Loss: 537.3705 mae: 26.526931762695312 rel. error: 0.11166811734437943  time : 8.425718907266855
>
>200 Loss: 416.01135 mae: 23.483247756958008 rel. error: 0.10002146661281586  time : 13.059221280738711
>
>300 Loss: 266.9216 mae: 21.457313537597656 rel. error: 0.09192200750112534  time : 17.616111068055034
>
>400 Loss: 239.48528 mae: 20.5916690826416 rel. error: 0.0883064866065979  time : 22.172612854279578
>
>500 Loss: 267.86072 mae: 20.23228645324707 rel. error: 0.0869283452630043  time : 26.740353140980005
>
>600 Loss: 222.4536 mae: 19.401844024658203 rel. error: 0.08398811519145966  time : 31.336414461024106
>
>700 Loss: 180.66728 mae: 19.385501861572266 rel. error: 0.08376768231391907  time : 35.92358886078
>
>800 Loss: 222.31805 mae: 18.806110382080078 rel. error: 0.08121315389871597  time : 40.523882042616606
>
>900 Loss: 140.94621 mae: 17.11213493347168 rel. error: 0.07491124421358109  time : 45.10136076621711
>
>1000 Loss: 102.3436 mae: 17.200101852416992 rel. error: 0.07441579550504684  time : 49.67014822270721
>
>1100 Loss: 213.91618 mae: 18.63749122619629 rel. error: 0.07999488711357117  time : 54.269442210905254
>
>1200 Loss: 198.0504 mae: 18.650419235229492 rel. error: 0.08025692403316498  time : 58.86295662727207
>
>1300 Loss: 91.117004 mae: 16.932586669921875 rel. error: 0.0732589140534401  time : 63.455598548054695
>
>1400 Loss: 115.33469 mae: 17.047874450683594 rel. error: 0.07385049015283585  time : 68.03953194618225
>
>1500 Loss: 188.22157 mae: 18.500423431396484 rel. error: 0.07881717383861542  time : 72.63241504132748
>
>1600 Loss: 73.99296 mae: 17.00739097595215 rel. error: 0.07307520508766174  time : 77.20965842809528
>
>1700 Loss: 119.67044 mae: 16.977487564086914 rel. error: 0.0728941336274147  time : 81.77187423035502
>
>1800 Loss: 118.60206 mae: 16.82490348815918 rel. error: 0.07225397974252701  time : 86.3691204348579
>
>1900 Loss: 74.22315 mae: 16.331043243408203 rel. error: 0.07074788212776184  time : 90.92316508665681
>
>2000 Loss: 69.09277 mae: 16.455432891845703 rel. error: 0.0706830695271492  time : 95.49115407746285
>
>2100 Loss: 169.36139 mae: 18.143535614013672 rel. error: 0.0776057243347168  time : 100.05525787267834
>
>2200 Loss: 105.73973 mae: 16.141504287719727 rel. error: 0.06914427876472473  time : 104.67397096380591
>
>2300 Loss: 112.43592 mae: 16.253826141357422 rel. error: 0.07009757310152054  time : 109.217095464468
>
>2400 Loss: 136.64384 mae: 17.38785743713379 rel. error: 0.07459509372711182  time : 113.78008683677763
>
>2500 Loss: 102.73479 mae: 16.36716079711914 rel. error: 0.07012166827917099  time : 118.3827364910394
>
>2600 Loss: 66.67031 mae: 16.380956649780273 rel. error: 0.07002332806587219  time : 122.9673305992037
>
>2700 Loss: 133.58382 mae: 17.387420654296875 rel. error: 0.07414235919713974  time : 127.65487424191087
>
>2800 Loss: 140.5376 mae: 17.233060836791992 rel. error: 0.07426299154758453  time : 132.3707270808518
>
>2900 Loss: 128.65556 mae: 16.50082015991211 rel. error: 0.07083127647638321  time : 137.0902198823169
>
>3000 Loss: 103.58169 mae: 16.51291275024414 rel. error: 0.07087258249521255  time : 141.79305426962674
>
>3100 Loss: 230.5178 mae: 17.801509857177734 rel. error: 0.07689779251813889  time : 146.44925714749843
>
>3200 Loss: 50.446766 mae: 15.771015167236328 rel. error: 0.06772685796022415  time : 150.99709218740463
>
>3300 Loss: 115.650055 mae: 16.614469528198242 rel. error: 0.0710296630859375  time : 155.58328399900347
>
>3400 Loss: 144.73999 mae: 18.16712188720703 rel. error: 0.07756192237138748  time : 160.2031332720071
>
>3500 Loss: 77.43564 mae: 16.369903564453125 rel. error: 0.07075335085391998  time : 164.77974649332464
>
>3600 Loss: 44.42608 mae: 15.697988510131836 rel. error: 0.06757614761590958  time : 169.35493067558855
>
>3700 Loss: 68.77671 mae: 16.144540786743164 rel. error: 0.06958482414484024  time : 173.91690846160054
>
>3800 Loss: 125.346436 mae: 17.08391571044922 rel. error: 0.07286818325519562  time : 178.4751351652667
>
>3900 Loss: 127.73309 mae: 17.198808670043945 rel. error: 0.07368797063827515  time : 183.0311270095408
>
>4000 Loss: 132.26308 mae: 16.717212677001953 rel. error: 0.0713069960474968  time : 187.59608391765505
>
>4100 Loss: 72.325836 mae: 15.670525550842285 rel. error: 0.06773883104324341  time : 192.1904318574816
>
>4200 Loss: 105.53636 mae: 19.256271362304688 rel. error: 0.08139962702989578  time : 196.8051517298445
>
>4300 Loss: 62.391106 mae: 15.962382316589355 rel. error: 0.0688060075044632  time : 201.39471916668117
>
>4400 Loss: 138.56642 mae: 17.031509399414062 rel. error: 0.07295973598957062  time : 205.9722465183586
>
>4500 Loss: 90.751175 mae: 16.03856086730957 rel. error: 0.06865857541561127  time : 210.55927829351276
>
>4600 Loss: 82.98366 mae: 15.691493034362793 rel. error: 0.06760484725236893  time : 215.12370301876217
>
>4700 Loss: 145.92674 mae: 17.186283111572266 rel. error: 0.07383159548044205  time : 219.68158520944417
>
>4800 Loss: 90.90707 mae: 16.00307273864746 rel. error: 0.06884384155273438  time : 224.25662205833942
>
>4900 Loss: 80.56631 mae: 15.907496452331543 rel. error: 0.0682239755988121  time : 228.83588003460318
>
>5000 Loss: 32.029076 mae: 15.157988548278809 rel. error: 0.06545406579971313  time : 233.40768042951822

## References
[1] Sepanta Zeighami, Ritesh Ahuja, Gabriel Ghinita, and Cyrus Shahabi, “A neural database for differentially private spatialrange queries, arXiv preprint: https://arxiv.org/abs/2108.01496

[2] Michal Piorkowski, Natasa Sarafijanovic-Djukic, and Matthias Grossglauser. 2009.CRAWDAD data set epfl/mobility (v. 2009-02-24)

[3] Eunjoon Cho, Seth A Myers, and Jure Leskovec. 2011. Friendship and mobility:user movement in location-based social networks. In Proceedings of the 17thACM SIGKDD international conference on Knowledge discovery and data mining.1082–1090
