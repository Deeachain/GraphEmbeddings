# Graph Embedding Study
## Support
- [x] Deepwalk
- [x] Node2Vec
## How to Run
### 1.Install Requirements
Create a virtual environment `conda create -n python36 python==3.6.5`, then `pip install -r requirements.txt`
### 2.Run
`python main.py`

### 3.Datasets
| Datasets | cora | karate | wiki  |
| :------: | :--: | :----: | :---: |
|    V     | 2708 |   34   | 2405  |
|    E     | 5429 |  156   | 17981 |
|  Class   |  7   |   -    |  17   |

## Visualization
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/deepwalk_cora_edges.png" width="450"><div align = "center">Deepalk_Cora</div></td>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/line_cora_edges.png" width="450”><div align = "center">Line_Cora</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/node2vec_cora_edges.png" width="450"><div align="center">Node2Vec_Cora</div></center></td>
    </tr>
</table>

|                      |  cora  |  wiki  |
| :------------------: | :----: | :----: |
| Deepwalk（f1-micro） | 0.8524 | 0.7131 |
|   Line（f1-micro）   |   -    |   -    |
| Node2vec（f1-micro） | 0.8505 | 0.7027 |



## Updating
- [ ] Line
The visualization of Embeddings study by Line is terrible,I'm fixing....
## Reference
[1] [Deepwalk Source code](https://github.com/phanein/deepwalk)  
[2] [Line Code]()  
[3] [Node2Vec Source code](https://github.com/aditya-grover/node2vec)  