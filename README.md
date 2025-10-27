

# Yelp Datasets


The YelpChi, YelpNyc, YelpZip datasets are obtained from the Yelp platform and are classified based on Yelp's filtering algorithm
and obtained from [Collective Opinion Spam Detection](https://dl.acm.org/doi/pdf/10.1145/2783258.2783370). 


The dataset contains user\_id / review\_id, product\_id, star rating, label, date, review\_content.

Products containing more than 800 reviews are removed to restrict the size of the computation graph [ [GraphConsis](https://arxiv.org/pdf/2005.00625) , [CareGNN](https://arxiv.org/pdf/2008.08692) ] to obtain the YelpChi dataset.
The same process has been followed for the much larger datasets of YelpNyc, YelpZip and the filtering threshold can be 
adjusted as needed. The statistics of the datasets before and after filtering are shown below. 



| Dataset     | #Reviews| #Users  | #Products | #Spam (\%)         | #Non-Spam (\%)         | #Spam Users (\%)     | #Non-Spam Users (\%)         |
|-------------|---------|---------|-----------|--------------------|------------------------|----------------------|------------------------------|
| **YelpChi** | 67,395  | 38,063  | 201       | 8,919  (13.2\%)    | 58,476 (86.7\%)        | 7,739  (20.3\%)      | 30,459   (80.0\%)            |
| **YelpNyc** | 359,022 | 160,225 | 923       | 36,855 (10.3\%)    | 322,167 (89.7\%)       | 28,496 (17.8\%)      | 132,386  (82.6\%)            |       
| **YelpZip** | 608,598 | 260,277 | 5,044     | 80,466 (13.2\%)    | 528,132 (86.8\%)       | 62,228 (23.9\%)      | 200,170  (76.9\%)            |
|  |  |  | | **After Filtering**  |  |  |  |
| **YelpChi** | 45,954  | 29,431  | 182       | 6,677  (14.5\%)    | 39,277  (85.5\%)       | 6,038  (20.5\%)      | 23,482  (79.8\%)             |
| **YelpNyc** | 191,398 | 104,460 | 812       | 19,824 (10.4\%)    | 171,574 (89.6\%)       | 16,917 (16.2\%)      | 87,831  (84.0\%)             |
| **YelpZip** | 414,135 | 200,935 | 4,912     | 59,216 (14.3\%)    | 354,919 (85.7\%)       | 48,224 (24.0\%)      | 154,271 (76.8\%)             |


It is to be noted that some users are present in both spam, non-spam categories.

## Graph Construction

### **Yelp** : Reviews as nodes and three relations: 

1) ***R-U-R*** : Connects reviews posted by the same user; 
2) ***R-S-R*** : Connects reviews under the same product with the same star rating (1-5 stars); 
3) ***R-T-R*** : Connects two reviews under the same product posted in the same month [two consecutive months].

The number of edges belonging to each relation is shown in table below. 

In the table below, the first **YelpChi** dataset is directly obtained from [`CareGNN`](https://github.com/YingtongDou/CARE-GNN/tree/master/data).  
The next three datasets are the replicated versions.


| Dataset      | #Nodes (Fraud\%)   | Relation 	 | #Edges         | Avg. Feature Similarity | Avg. Label Similarity|
|--------------|---------------------|---------  |----------------|-------------------------|----------------------|
| **YelpCHI-1** | 45,954 (14.5\%)   | *R-U-R*    | 49,315         | 0.991                   | 0.909                |
|             |                     | *R-T-R*    | 573,616        | 0.988                   | 0.176                |
|             |                     | *R-S-R*    | 3,402,743      | 0.988                   | 0.186                |
|             |                     | *ALL*      | 3,846,979      | 0.988                   | 0.184                |
| **YelpCHI** | 45,954 (14.5\%)     | *R-U-R*    | 49,315         | 0.988                   | 0.998                |
|             |                     | *R-T-R*    | 573,616        | 0.985                   | 0.199                |
|             |                     | *R-S-R*    | 3,402,743      | 0.985                   | 0.197                |
|             |                     | *ALL*      | 3,846,979      | 0.985                   | 0.197                |
| **YelpNYC** | 191,398 (10.4\%)    | *R-U-R*    | 498,732        | 0.989                   | 0.580                |
|             |                     | *R-T-R*    | 2,379,443      | 0.986                   | 0.164                |
|             |                     | *R-S-R*    | 12,214,234     | 0.986                   | 0.115                |
|             |                     | *ALL*      | 14,252,338     | 0.986                   | 0.136                |
| **YelpZIP** | 414,135 (14.3\%)    | *R-U-R*    | 1,558,877      | 0.989                   | 0.502                |
|             |                     | *R-T-R*    | 3,638,191      | 0.985                   | 0.191                |
|             |                     | *R-S-R*    | 19,437,968     | 0.985                   | 0.150                |
|             |                     | *ALL*      | 23,379,486     | 0.986                   | 0.175                |


The similarity scores need to be calculated based on [`simi_comp.py`](https://github.com/YingtongDou/CARE-GNN/blob/master/simi_comp.py).

**Features**

For the node features, 32 handcrafted features from [Collective Opinion Spam Detection](https://dl.acm.org/doi/pdf/10.1145/2783258.2783370) are taken. 
15 Review, 9 User, 8 Product features are concatenated for each review node. Reviews from same user/product have the same user/product features.

The features are spam priors calculated based on Sec. 2.2.1 of the above paper.

The MatLab code to obtain the features is [here](https://www.dropbox.com/scl/fo/brfrsg6w7s1olf5ce9ebd/AFT9FJm-7-xvg1s-P3z1slw?rlkey=zzrh34yc3dd34fug2ewcxn8lf&e=1&dl=0).


## Datasets

The .mat, .dgl files of the three datasets are available in the data folder. The code to obtain the datasets is in dataset_mat.py , dataset_dgl.py. The .mat files are generated first and then transformed into a .dgl format.


