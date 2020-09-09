# Graph Embedding Study
## Support
- [x] Deepwalk
- [x] Node2Vec
## How to Run
### 1.Install Requirements
Create a virtual environment `conda create -n python36 python==3.6.5`, then `pip install requirements.txt`
### 2.Run
`python main.py`

### 3.Datasets
| Datasets | cora | karate | wiki  |
| :------: | :--: | :----: | :---: |
|    V     | 2708 |   34   | 2405  |
|    E     | 5429 |  156   | 17981 |
|  Class   |  7   |   -    |  17   |

## Visualization

|                      |  cora  |  wiki  |
| :------------------: | :----: | :----: |
| Deepwalk（f1-micro） | 0.8524 | 0.7131 |
|   Line（f1-micro）   |   -    |   -    |
| Node2vec（f1-micro） | 0.8505 | 0.7027 |



## Updating
- [ ] Line
## Reference
[1] [Deepwalk Source code](https://github.com/phanein/deepwalk)  
[2] [Line Code]()  
[3] [Node2Vec Source code](https://github.com/aditya-grover/node2vec)  