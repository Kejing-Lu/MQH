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
## Parameters

* See the explanations of parameters in run_example.sh
