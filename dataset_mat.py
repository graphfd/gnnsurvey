# -*- coding: utf-8 -*-

"""
This program reads the metadata, reviewContent files and prodFeatures, userFeatures, reviewFeatures feature files 
Filters all the files based on prod having > 800 reviews 
Builds the graph from the filtered metadata and saves it to .mat file

The following changes need to be made for YelpNyc, YelpZip :


line 43,44

metadata_Yelp       =  'metadata_YelpChi'        metadata_YelpChi/Nyc/Zip
reviewContent_Yelp  =  'reviewContent_YelpChi'

line 73-75 : 

prodFeatures   = 'z_YelpChi_priorP.txt'
userFeatures   = 'z_YelpChi_priorU.txt'
reviewFeatures = 'z_YelpChi_priorR.txt'


line 189 : metadata_filtered_file.to_csv('metadata_filtered_file_YelpChi')

line 507 : metadata_filtered_file_path = 'metadata_filtered_file_YelpChi'
"""



import pandas as pd
import csv, os, gzip
import numpy as np
import copy as cp
import pandas as pd
import scipy.sparse as sp

from datetime import datetime
from scipy.io import loadmat, savemat 
from scipy.sparse import csr_matrix



metadata_Yelp       =  'metadata_YelpChi'       # metadata, reviewContent files of YelpChi
reviewContent_Yelp  =  'reviewContent_YelpChi'


with open(reviewContent_Yelp, 'r') as file:
    lines = file.readlines()

lines = [line.strip() for line in lines]      # Strip newline characters and any extra spaces
lines = [line.split('\t') for line in lines]  # separate into columns 


metadata      =  pd.read_csv( metadata_Yelp, sep=' ', names=['user_id', 'prod_id', 'rating', 'label', 'date'] )
reviewContent =  pd.DataFrame(lines, columns=['user_id', 'prod_id', 'date', 'review'])

reviewContent['user_id'] = reviewContent['user_id'].astype(int)
reviewContent['prod_id'] = reviewContent['prod_id'].astype(int)



user_features   = ['MNR_user', 'PR_user', 'NR_user', 'avgRD_user', 'WRD_user', 'ERD_user', 'BST_user', 
                   'ETG_user', 'RL_user', 'ACS_user', 'MCS_user' ]
        
prod_features   = ['MNR_prod', 'PR_prod', 'NR_prod', 'avgRD_prod', 'WRD_prod', 'ERD_prod', 
                   'ETG_prod', 'RL_prod', 'ACS_prod', 'MCS_prod' ]

review_features = ['Rank_reviews', 'RD_reviews', 'EXT_reviews', 'DEV_reviews', 'ETF_reviews', 'ISR_reviews',
                   'PCW_reviews', 'PC_reviews', 'L_reviews', 'PP1_reviews', 'RES_reviews',
                   'SW_reviews', 'OW_reviews', 'F_reviews', 'DL_u', 'DL_b' ]


prodFeatures   = 'z_YelpChi_priorP.txt'
userFeatures   = 'z_YelpChi_priorU.txt'
reviewFeatures = 'z_YelpChi_priorR.txt'


prodFeatures    =  pd.read_csv( prodFeatures,   sep=' ',  names=prod_features )
userFeatures    =  pd.read_csv( userFeatures,   sep=' ',  names=user_features )
reviewFeatures  =  pd.read_csv( reviewFeatures, sep=' ',  names=review_features )



# Remove specified columns
prodFeatures_cg    =  prodFeatures.drop(columns=['ACS_prod', 'MCS_prod'])
userFeatures_cg    =  userFeatures.drop(columns=['ACS_user', 'MCS_user'])
reviewFeatures_cg  =  reviewFeatures.drop(columns=['F_reviews'])



# Rearrange columns in userFeatures
user_columns =  userFeatures_cg.columns.tolist()
erd_index    =  user_columns.index('ERD_user')
bst_index    =  user_columns.index('BST_user')

user_columns[erd_index], user_columns[bst_index] = user_columns[bst_index], user_columns[erd_index]
userFeatures_cg  =  userFeatures_cg[user_columns]


metadata_cg       =  metadata.copy()
reviewContent_cg  =  reviewContent.copy()


'''
The features file from MatLab code has index which starts from 0, 
the metadata has user_id from 201, so, the metadata user_id needs to be re-seeded to start from 0.
'''

unique_users   =  metadata_cg['user_id'].unique()
user_id_map    =  {old_id: new_id for new_id, old_id in enumerate(unique_users)}


metadata_cg['user_id']       =  metadata_cg['user_id'].map(user_id_map)
reviewContent_cg['user_id']  =  reviewContent_cg['user_id'].map(user_id_map)


'''
metadata_cg : user_id  prod_id  rating  label  date ; reviewContent_cg : user_id  prod_id  date  review ; 
userFeatures_cg : user_features ; prodFeatures_cg : prod_features ; reviewFeatures_cg : review_features ;

while merging the index itself is taken as the id to merge on if left/right_on is not specified 
'''


user_feat_app   =   metadata_cg.merge(    userFeatures_cg,    left_on='user_id',   right_index=True  ) 

prod_feat_app   =   user_feat_app.merge(  prodFeatures_cg,    left_on='prod_id',   right_index=True  ) 

full_features   =   prod_feat_app.merge(  reviewFeatures_cg,  left_index=True,     right_index=True  )


rev_feat_app   =   reviewFeatures_cg.merge( prod_feat_app,   left_index=True,   right_index=True  )

reviewFeatures_cg_final  =  rev_feat_app.drop(  columns=['user_id', 'prod_id', 'rating', 'label', 'date']  )



# code removing prod having > 800 reviews and their reviews, users 

prod            =  reviewContent['prod_id'].value_counts()
prod_count_800  =  [ (prod_id, count)  for prod_id, count in prod.items() if count > 800 ] 
prod_800        =  [ prod_id           for prod_id, count in prod.items() if count > 800 ] 


prod_counts = reviewContent['prod_id'].value_counts()

#  prod_id values that appear more than 800 times
prod_ids_to_remove = prod_counts[prod_counts > 800].index.astype(int)

# Remove rows from metadata and reviewContent where prod_id appears in prod_ids_to_remove
# Make sure the prod_id data type is same in metadata, reviewContent,  prod_ids_to_remove

metadata_fltd       =  full_features[~full_features['prod_id'].isin(prod_ids_to_remove)]
reviewContent_fltd  =  reviewContent_cg[~reviewContent_cg['prod_id'].isin(prod_ids_to_remove)]


metadata_fltd['prod_id'].value_counts()  # 182
metadata_fltd['user_id'].value_counts()  # 29431
reviewContent_fltd['prod_id'].value_counts()      # 182

# end of removing code


# After filtering, the remaining users/reviews are 29431, products are 182, so they need to be reseeded 


metadata_filtered       =  metadata_fltd.copy().reset_index(drop=True)
reviewContent_filtered  =  reviewContent_fltd.copy().reset_index(drop=True)


unique_users   =  metadata_filtered['user_id'].unique()
user_id_map    =  {old_id: new_id for new_id, old_id in enumerate(unique_users)}

metadata_filtered['user_id']       =  metadata_filtered['user_id'].map(user_id_map)
reviewContent_filtered['user_id']  =  reviewContent_filtered['user_id'].map(user_id_map)



unique_prods   =  metadata_filtered['prod_id'].unique()
prod_id_map    =  {old_id: new_id for new_id, old_id in enumerate(unique_prods)}

metadata_filtered['prod_id']       =  metadata_filtered['prod_id'].map(prod_id_map)
reviewContent_filtered['prod_id']  =  reviewContent_filtered['prod_id'].map(prod_id_map)



metadata_filtered_file = metadata_filtered[['user_id', 'prod_id', 'rating', 'label', 'date']]

metadata_filtered_file.to_csv('metadata_filtered_file_YelpChi', sep=' ', header=False , \
                               index=False, quoting=csv.QUOTE_NONE)


metadata_filtered_file['label'].value_counts()     # 1 : 39277  ,  -1 : 6677   
        
metadata_filtered_file['label'].value_counts(1)    # 1 : 0.8547  ,  -1 : 0.1453





remove_columns =  ['user_id', 'prod_id', 'rating', 'label', 'date']

keep_columns   =  [ 
  'Rank_reviews', 'RD_reviews', 'EXT_reviews', 'DEV_reviews', 'ETF_reviews', 'ISR_reviews', 'PCW_reviews', 
  'PC_reviews', 'L_reviews', 'PP1_reviews', 'RES_reviews', 'SW_reviews', 'OW_reviews', 'DL_u', 'DL_b',        
  'MNR_user', 'PR_user', 'NR_user', 'avgRD_user', 'WRD_user', 'BST_user', 'ERD_user', 'ETG_user', 'RL_user', 
  'MNR_prod', 'PR_prod', 'NR_prod', 'avgRD_prod', 'WRD_prod', 'ERD_prod', 'ETG_prod', 'RL_prod', 
  ]

reviewFeatures_cg_filtered = metadata_filtered.drop(columns=remove_columns).reindex(columns=keep_columns)




# Building the graph from the metadata 

def read_graph_data(metadata_filename, graph_data_structure='up'):
    """ 
            Read the user-review-product graph from file. Can output the graph in different formats

            Args:
                    metadata_filename: a gzipped file containing the graph.
                    graph_data_structure: defines the output graph format
                            'up' (default) ---> user-product and product-user graphs
                            'urp' ---> user-review and review-product graphs
            Return:
                    graph1: user-product / user-review
                    graph2: product-user / review-product
    """

    user_data = {}

    prod_data = {}

    # use the rt mode to read ascii strings instead of binary
    with open(metadata_filename, 'rt') as f:
        # file format: each line is a tuple (user id, product id, rating, label, date)
        for line in f:
            items = line.strip().split()
            u_id = items[0]
            p_id = items[1]
            rating = float(items[2])
            label = int(items[3])
            date = items[4]

            if u_id not in user_data:
                user_data[u_id] = []
            user_data[u_id].append((p_id, rating, label, date))

            if p_id not in prod_data:
                prod_data[p_id] = []
            prod_data[p_id].append((u_id, rating, label, date))

    # read text feature files, including: wordcount, ratio of SW/OW, etc.
    # constructed by the python files provided by the authors.
    
    print('read_graph_data')
    print('read reviews from %s' % metadata_filename)
    print('number of users = %d' % len(user_data))
    print('number of products = %d' % len(prod_data))

    if graph_data_structure == 'up':
        return user_data, prod_data

    if graph_data_structure == 'urp':
        user_review_graph = {}
        for k, v in user_data.items():
            user_review_graph[k] = []
            for t in v:
                # (u_id, p_id) representing a review
                user_review_graph[k].append((k, t[0]))
        review_product_graph = {}
        for k, v in prod_data.items():
            for t in v:
                # (u_id, p_id) = (t[0], k) is the key of a review
                review_product_graph[(t[0], k)] = k
        return user_review_graph, review_product_graph


def remove_reviews(upg, pug):

    # In prod_user_graph, removes products which have more than 800 reviews 

    # In user_prod_graph, removes reviews whose products have been removed in prod_user_graph
    # In user_prod_graph, removes users   whose reviews count == 0 , i.e. users with no reviews
    
    print('remove_reviews start')

    user_prod_graph = cp.deepcopy(upg)
    prod_user_graph = cp.deepcopy(pug)

    removed_prod = []
    removed_user = []
    for prod, reviews in prod_user_graph.items():
        if len(reviews) > 800:
            removed_prod.append(prod)

    for user, reviews in user_prod_graph.items():
        for review in reviews:
            if review[0] in removed_prod:
                user_prod_graph[user].remove(review)

    for user, reviews in user_prod_graph.items():
        if len(reviews) == 0:
            removed_user.append(user)

    return removed_user, removed_prod


def load_new_graph(metadata_filename, removed_user, removed_prod):

    # Removes the user and product information obtained from remove_reviews()

    user_data = {}
    prod_data = {}

    # use the rt mode to read ascii strings instead of binary
    with open(metadata_filename, 'rt') as f:
        # file format: each line is a tuple (user id, product id, rating, label, date)
        for line in f:
            items = line.strip().split()
            u_id = items[0]
            p_id = items[1]
            rating = float(items[2])
            label = int(items[3])
            date = items[4]

            if u_id not in removed_user and p_id not in removed_prod:
                if u_id not in user_data:
                    user_data[u_id] = []
                user_data[u_id].append((p_id, rating, label, date))

                if p_id not in prod_data:
                    prod_data[p_id] = []
                prod_data[p_id].append((u_id, rating, label, date))

    # read text feature files, including: wordcount, ratio of SW/OW, etc.
    # constructed by the python files provided by the authors.
    
    print('load_new_graph start')
    
    print('read reviews from %s' % metadata_filename)
    print('number of users = %d' % len(user_data))
    print('number of products = %d' % len(prod_data))

    return user_data, prod_data


def create_ground_truth(user_data):
    """
    Given user data, return a dictionary of labels of users and reviews

    Args:
            user_data: key = user_id, value = list of review tuples.

    Return:
            user_ground_truth   : key = user id (not prefixed), value = 0 (non-spam) /1 (spam) 
            review_ground_truth :     review id (not prefixed), value = 0 (non-spam) /1 (spam) 
    """
    user_ground_truth   =  {}
    review_ground_truth =  {}
    

    for user_id, reviews in user_data.items():

        user_ground_truth[user_id] = 0

        for r in reviews:
            prod_id = r[0]
            label = r[2]

            if label == -1:
                review_ground_truth[(user_id, prod_id)] = 1
                user_ground_truth[user_id] = 1
            else:
                review_ground_truth[(user_id, prod_id)] = 0

    return user_ground_truth, review_ground_truth


def time_judge(time1, time2):

    date1 = datetime.strptime(time1, '%Y-%m-%d')
    date2 = datetime.strptime(time2, '%Y-%m-%d')

    if date1.year == date2.year and abs(date1.month - date2.month) <= 1 :
        return True
    else:
        return False



def meta_to_homo(meta_data_name):
    """
    generating homogeneous adjacency matrix from metadata
    :return:
    """

    upg, pug = read_graph_data(meta_data_name)

    removed_user, removed_prod = remove_reviews(upg, pug)

    user_prod_graph, prod_user_graph = load_new_graph( meta_data_name, removed_user, removed_prod )

    user_ground_truth, review_ground_truth = create_ground_truth( user_prod_graph )

    # map review id to adj matrix id
    rid_mapping = {}
    r_index = 0
    for review in review_ground_truth.keys():
        rid_mapping[review] = r_index
        r_index += 1

    review_adj = sp.lil_matrix( (len(review_ground_truth), len(review_ground_truth)) )
    
    review_adj_rur = sp.lil_matrix( (len(review_ground_truth), len(review_ground_truth)) )
    review_adj_rtr = sp.lil_matrix( (len(review_ground_truth), len(review_ground_truth)) )
    review_adj_rsr = sp.lil_matrix( (len(review_ground_truth), len(review_ground_truth)) )
    # review_adj_rstr = sp.lil_matrix( (len(review_ground_truth), len(review_ground_truth)) )

    # review-review graph with stacked multiple relations

    '''
	
	nodes : reviews , relations : r-product-r, r-user-r, r-time-r, r-star-r, r-star/time-r 


    r-product-r   : Connects reviews of the same product 
    r-user-r      : Connects reviews made by the same user 
    r-time-r      : Connects reviews for the  same product  and are time-related [same month]
    r-star-r      : Connects reviews for the  same product  if they have the same rating
    r-star/time-r : Connects reviews for the  same product  if they have the same rating and are time-related [same month]

    
    YelpCHI : Reviews as nodes and three relations : R-U-R, R-S-R, R-T-R 

    1) R-U-R: it connects reviews posted by the same user; 
    2) R-S-R: it connects reviews under the same product with the same star rating (1-5 stars); 
    3) R-T-R: it connects two reviews under the same product posted in the same month. 

	'''
    
    # user_prod_graph : user_data[u_id] = ( p_id, rating, label, date )   

    # prod_user_graph : prod_data[p_id] = ( u_id, rating, label, date )


    # # 1) r-product-r
    # for p, reviews in prod_user_graph.items():
    #     for r0 in reviews:
    #         for r1 in reviews:
    #             if r0[0] != r1[0]:  # do not add self loop at this step
    #                 review_adj[rid_mapping[(r0[0], p)],
    #                            rid_mapping[(r1[0], p)]] = 1
    
    print("r-user-r start")
    
    # 2) r-user-r    reviews posted by the same user
    for u, reviews in user_prod_graph.items():
        for r0 in reviews:
            for r1 in reviews:
                if r0[0] != r1[0]:
                    review_adj[rid_mapping[(u, r0[0])],
                               rid_mapping[(u, r1[0])]] = 1
                    review_adj_rur[rid_mapping[(u, r0[0])],
                                  rid_mapping[(u, r1[0])]] = 1
    
    print("r-time-r start")
    
    # 3) r-time-r    reviews under the same product posted in the same month
    for p, reviews in prod_user_graph.items():
        for r0 in reviews:
            for r1 in reviews:
                if r0[0] != r1[0] and time_judge(r0[3], r1[3]) == True:
                    review_adj[rid_mapping[(r0[0], p)],
                               rid_mapping[(r1[0], p)]] = 1
                    review_adj_rtr[rid_mapping[(r0[0], p)],
                                   rid_mapping[(r1[0], p)]] = 1

    print("r-star-r start") 

    # 4) r-star-r    reviews under the same product with the same rating
    for p, reviews in prod_user_graph.items():
        for r0 in reviews:
            for r1 in reviews:
                if r0[0] != r1[0] and r0[1] == r1[1]:
                    review_adj[rid_mapping[(r0[0], p)],
                               rid_mapping[(r1[0], p)]] = 1
                    review_adj_rsr[rid_mapping[(r0[0], p)],
                                   rid_mapping[(r1[0], p)]] = 1
                    
    print("r-star-r done")                    

    #  # 5) r-star/time-r
    # for p, reviews in prod_user_graph.items():
    #      for r0 in reviews:
    #          for r1 in reviews:
    #              if r0[0] != r1[0] and r0[1] == r1[1] and time_judge(r0[3], r1[3]) == True:
    #                  review_adj[rid_mapping[(r0[0], p)],
    #                             rid_mapping[(r1[0], p)]] = 1
    #                  review_adj_rstr[rid_mapping[(r0[0], p)],
    #                             rid_mapping[(r1[0], p)]] = 1
    
    return review_adj, review_adj_rur, review_adj_rtr, review_adj_rsr, review_ground_truth


metadata_filtered_file_path = 'metadata_filtered_file_YelpChi'


# generate homogeneous adjacency matrix
review_adj, review_adj_rur, review_adj_rtr, review_adj_rsr, review_ground_truth  =  meta_to_homo(metadata_filtered_file_path)



# Different from previous program where the labels were directly obtained from metadata file and 
# as a result were in a different order wrt to the output of meta_to_homo() -> create_ground_truth() 

# review_ground_truth   #   [ (user_id, prod_id) : (label) ]


review_labels  =  np.array(list(review_ground_truth.values()))


# (user_id, prod_id) is the unique identifier of each review as in meta_to_homo() 
review_id = [ (int(k[0]), int(k[1])) for k in review_ground_truth ]



# here the metadata_filtered is before meta_to_homo() and review_id is obtained from meta_to_homo() 
# hence creating a dictionary from metadata_filtered with ['user_id', 'prod_id'] as the key

node_features_dict = {  (row['user_id'], row['prod_id']): ( row['MNR_user':'DL_b'].values )
                        for _, row in metadata_filtered.iterrows()   }


missing_keys = [key for key in review_id if key not in node_features_dict]

if missing_keys: print(f"Warning: {len(missing_keys)} keys missing from node_features_dict")


# the order of labels is from review_ground_truth and reviews [features] is from the review_id 
# which is obtained from review_ground_truth; hence both [labels, features] have same order

node_features = np.array([ node_features_dict[key] for key in review_id if key in node_features_dict ] , dtype=float) 

node_features_sp  =  csr_matrix(node_features)




review_adj_sparse = review_adj.tocsr() 

review_adj_rur_sparse = review_adj_rur.tocsr() 
review_adj_rtr_sparse = review_adj_rtr.tocsr() 
review_adj_rsr_sparse = review_adj_rsr.tocsr() 
# review_adj_rstr_sparse = review_adj_rstr.tocsr()


# number of edges of each type 
review_adj_sparse.nnz/2
review_adj_rur_sparse.nnz/2
review_adj_rtr_sparse.nnz/2
review_adj_rsr_sparse.nnz/2



# Step 3: Save the graph and features to a .mat file
data = {
    'homo'       :     review_adj_sparse,
    'net_rur'    :     review_adj_rur_sparse,
    'net_rtr'    :     review_adj_rtr_sparse,
    'net_rsr'    :     review_adj_rsr_sparse,
    'features'   :     node_features_sp,
    'label'      :     review_labels
}



# Save to .mat file
# savemat('YelpChi.mat', data)

# loadmat('YelpChi.mat')


