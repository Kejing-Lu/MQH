# Project Title

MQH: Locality Sensitive Hashing on Multi-level Quantization Errors for Point-to-Hyperplane Distances

## Getting Started

### Datasets

* Music, Tiny1M, Glove: https://drive.google.com/drive/folders/1aBFV4feZcLnQkDR7tjC-Kj7g3MpfBqv7
* Deep: https://disk.yandex.ru/d/11eDCm7Dsn9GA/base
(We extracted the first 10M and 100M from the base set. Query sets and ground_truth sets are attached in folders ./MQH/data/Deep10M/ and ./MQH/data/Deep100M. In addition, for the extracted data sets of Deep10M and Deep100M, the dimension of every vector should be extended from 96 to 97 with the 97-th dimension 1.)

### Compile On Ubuntu
Complie MQH

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```
## Run MQH (An example for Music) 
```shell
$ bash run_example.sh
```
## Description of Parameters
Paramters of Data 
* `n` is data size, `d` is the dimension of data and `q` is the size of query set.
* The path of data can be set directly in the provided script.

Internal Parameters
* `l0` is an internal parmeter and is suggested to set to 5. the number of traning samples is set to 10K.

User-specified Parameters
* `k` is the number of returned points (top-k search).
* `delta` is the parameter controlling the tradeoff between efficiency and accuracy. `delta` is suggested to take value in [0.3, 0.5]. 
* `flag` is for the choice of algorithm. `flag=1` for the algorithm with strict probability guarantees on recall rates and `flag=0` for the fast algorithm with conditional guarantees. 

