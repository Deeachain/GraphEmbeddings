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
| datasets | cora |  dblp  | BlogCatalog |
| :----: | :--: | :----: | :---------: |
|   V    | 2708 | 17725  |    10312    |
|   E    | 5429 | 105781 |   333983    |
| Class  |  7   |   4    |     39      |

### 4.NodeClassify Task
|                      |  cora  |  dblp  |
| :------------------: | :----: | :----: |
| Deepwalk（f1-micro） | 0.8542 | 0.8327 |
|   Line（f1-micro）   | 0.6218 | 0.6262 |
| Node2vec（f1-micro） | 0.8561 | 0.8386 |

## Visualization
<table>
    <tr>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/deepwalk_cora_edges.png"><div align = "center">Deepalk_Cora</div></td>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/line_cora_edges.png"><div align = "center">Line_Cora</div></center></td>
        <td ><center><img src="https://github.com/Deeachain/GraphEmbeddings/blob/master/output/visualization/node2vec_cora_edges.png"><div align="center">Node2Vec_Cora</div></center></td>
    </tr>
</table>

## Updating
- [ ] Line--The visualization of Embeddings study by Line is terrible,I'm fixing....

## Reference
[1] [Deepwalk Source code](https://github.com/phanein/deepwalk)  
[2] [Line Source Code C++](https://github.com/tangjianpku/LINE)  
[3] [Node2Vec Source code](https://github.com/aditya-grover/node2vec)  