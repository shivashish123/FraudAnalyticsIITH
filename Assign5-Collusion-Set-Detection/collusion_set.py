import numpy as np
import pandas as pd
from collections import defaultdict

# data = pd.read_csv('data.csv')
data = pd.read_csv('dataset_paper.csv')

print(len(data['Seller Id'].unique()))

print(len(data['Buyer ID'].unique()))

print(f'Size of data is {len(data)}')

seller_ids = data['Seller Id'].unique()
buyer_ids = data['Buyer ID'].unique()
# all_traders_ids = np.union1d(seller_ids, buyer_ids)
# traders who are both sellers and buyers can be involved in circular trading
trader_ids = np.intersect1d(seller_ids, buyer_ids)
trader_ids_set = set(list(trader_ids))

print(f'Number of traders: {len(trader_ids_set)}')

adj_list = defaultdict(dict)

for transaction_idx in range(len(data)):
    seller_id = data.iloc[transaction_idx]['Seller Id']
    buyer_id = data.iloc[transaction_idx]['Buyer ID']
    quantity = data.iloc[transaction_idx]['Anount in Lakhs']
    if seller_id not in trader_ids_set and buyer_id not in trader_ids_set:
        continue
    if buyer_id in adj_list[seller_id]:
        adj_list[seller_id][buyer_id] += quantity
    else:
        adj_list[seller_id][buyer_id] = quantity


# input k, m, h
k = 6
m = 1
h = 0.6

# retaining ony kNN for each node
for trader_id in adj_list.keys():
    if len(adj_list[trader_id]) <= k:
        continue
    adj_list[trader_id] = dict(sorted(adj_list[trader_id].items(), key=lambda item: -item[1]))
    kNN_neighbours = {key: adj_list[trader_id][key] for key in list(adj_list[trader_id].keys())[:k]}
    adj_list[trader_id] = kNN_neighbours

collusion_sets = [[trader_id] for trader_id in trader_ids]

def find_collusion_index(cluster):
    if len(cluster) == 1 and cluster[0] not in adj_list[cluster[0]]:
        return 0
    
    internal_trading = 0
    external_trading = 0
    
    cluster_set = set(cluster)
    
    for i in range(len(cluster)):
        # cluster[i] is seller, j is buyer
        for j in adj_list[cluster[i]]:
            if j in cluster_set:
                internal_trading += adj_list[cluster[i]][j]
            else:
                external_trading += adj_list[cluster[i]][j]
        
        # cluster[i] is buyer, j is seller
        for j in seller_ids:
            if j in cluster_set:
                continue
            if cluster[i] in adj_list[j]:
                external_trading += adj_list[j][cluster[i]]

    if external_trading == 0:
        return float('INF')
    collusion_index = internal_trading/external_trading
    return collusion_index


def find_collusion_level(cluster_1, cluster_2):
    union_cluster = np.union1d(cluster_1, cluster_2)
    collusion_index = find_collusion_index(union_cluster)
    return collusion_index


def find_collusion_levels():
    collusion_levels = []
    for i in range(len(collusion_sets)):
        for j in range(i+1, len(collusion_sets)):
            cluster_1 = collusion_sets[i]
            cluster_2 = collusion_sets[j]
            #print(cluster_1, cluster_2)
            collusion_level = find_collusion_level(cluster_1, cluster_2)
            collusion_levels.append([collusion_level, cluster_1, cluster_2])
    collusion_levels.sort(reverse=True)
    return collusion_levels


def check_point_compatibility(point, cluster):
    cluster_set = set(cluster)
    n_cluster = len(cluster)
    close_neighbours = 0
    for neighbour in adj_list[point]:
        if neighbour in cluster_set:
            close_neighbours += 1
    return (close_neighbours >= min(m, n_cluster))


# k, m, h compatiblity
def check_cluster_compatibility(cluster_1, cluster_2):
    n_points_1 = len(cluster_1)
    n_points_2 = len(cluster_2)
    
    compatible_points = 0
    for point in cluster_1:
        if check_point_compatibility(point, cluster_2):
            compatible_points += 1
    if compatible_points >= h*n_points_1:
        return True

while(True):
    print(f'Collusion set size: {len(collusion_sets)}')
    collusion_levels = find_collusion_levels()
#     print(f'Collusion levels size: {len(collusion_levels)}')
    compatible_pair_exists = False
    for i in range(len(collusion_levels)):
        #print(i, end=' ')
        cluster_1 = collusion_levels[i][1]
        cluster_2 = collusion_levels[i][2]
        collusion_level = collusion_levels[i][0]
        if collusion_level == 0:
            continue
        if (check_cluster_compatibility(cluster_1, cluster_2) and
            check_cluster_compatibility(cluster_2, cluster_1)):
            collusion_sets.remove(cluster_1)
            collusion_sets.remove(cluster_2)
            collusion_sets.append(cluster_1 + cluster_2)
            compatible_pair_exists = True
#             print(f'{cluster_1} and {cluster_2} are merged')
            break
    if not compatible_pair_exists:
        break

new_collusion_sets = [collusion_set for collusion_set in collusion_sets if len(collusion_set) != 1]


print(new_collusion_sets)





