# grarepid

***

### Installations and Setup

1. Create a conda environment or a python venv environment: <br>

``` conda create --name <my-env> ``` <br>
 ``` python3 -m venv <myenvname> ``` 

<br>
2. Install all the dependencies listed in ` requirements.txt ` file. <br>
3.   Load torch datasets correctly,
      Synthetic graph: ``` graph = torch.load('../random_graphs/sbm_torch_5_7_7') ``` <br>
      Real graph: ``` dataset = Planetoid(root='../real_graphs/planetoid/', name=’Cora’) ``` <br>  
      ``` Graph = dataset[0] ```

\
4. Execute NC-LID algorithm from ```./nclid/nclideval.py``` script. <br> Execute ```./id4geol/intrinsic_dimension_k_hops.py``` script for the GEOL algorithm. \
5. Execute ```./embeddings/driver_sage_gat_gcn.py``` script to obtain graph’s node embeddings and node classification results from graphSAGE, GAT and GCN algorithms. \
6. Execute ```./embeddings/driver_node2vec.py``` script to obtain node embeddings from Node2vec algorithm. \
7. Execute ```./downstream_tasks/link_prediction.py``` and ```./downstream_tasks/anomaly_detection.py``` to obtain results for Graph Machine Learning applications. 

8. Execute ```./embed_ids/embed_skdim.py``` script to obtain Intrinsic Dimensionalities of the computed Embeddings obtained in steps 5 and 6. \
9. Perform analysis on results obtained in steps 4, 7 and, 8. 


## Visuals

![alt text](results/plots_sbm/Correlation:_Graph_IDs_vs_Embedding_IDs_(mind_ml).png) 

![alt text](results/plots_sbm/Correlation:_Graph_IDs_vs_Node_Classification_Metrics.png) 

 ![alt text](results/plots_sbm/Correlation:_Graph_Metrics_vs_Node_Classification_Metrics.png) 

 ![alt text](./results/plots_sbm/Scatter_anomaly_avg_precision_score_anomaly_prediction__vs__dim_graph_geol.png) 

 ![alt text](./results/plots_sbm/Scatter_close_cent_graph_metrics__vs__mind_ml_sage_embeddings.png) 
