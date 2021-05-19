import numpy as np  
import pandas as pd
from collections import defaultdict

# reading dataset
data = pd.read_csv('data.csv')
data = data[:10000]

print(f'Size of data is {len(data)}')
seller_ids = data['Seller Id'].unique()
buyer_ids = data['Buyer ID'].unique()
# traders who are both sellers and buyers can be involved in circular trading
trader_ids = np.intersect1d(seller_ids, buyer_ids)
trader_ids_set = set(list(trader_ids))
print(f'Number of traders: {len(trader_ids_set)}')


# graph construction
adj_list = defaultdict(defaultdict)

for transaction_idx in range(len(data)):
    seller_id = data.iloc[transaction_idx]['Seller Id']
    buyer_id = data.iloc[transaction_idx]['Buyer ID']
    if seller_id not in trader_ids_set or buyer_id not in trader_ids_set:
        continue
    quantity = data.iloc[transaction_idx]['Anount in Lakhs']
    if buyer_id in adj_list[seller_id]:
        adj_list[seller_id][buyer_id] += quantity
    else:
        adj_list[seller_id][buyer_id] = quantity

# input k, m, h
k = 4
m = 1
h = 0.7

# retaining ony kNN for each node
for trader_id in adj_list:
    quantity = []
    if len(adj_list[trader_id]) <= k:
        continue
    for neighbour in list(adj_list[trader_id]):
        quantity.append(adj_list[trader_id][neighbour])
    quantity.sort(reverse=True)
    kth_largest_quantity = quantity[min(len(quantity), k)-1]
    for neighbour in list(adj_list[trader_id]):
        if adj_list[trader_id][neighbour] < kth_largest_quantity:
            del adj_list[trader_id][neighbour]


# for x, y in adj_list.items():
#     print(x, end=' ')
#     print(len(y))


collusion_sets = [[trader_id] for trader_id in trader_ids]


def find_collusion_index(cluster):
    if len(cluster) == 1 and cluster[0] not in adj_list[cluster[0]]:
        return 0
    internal_trading = 0
    external_trading = 0
    cluster_set = set(cluster)
    for i in range(len(cluster)):
        # internal trading
        for j in range(i+1, len(cluster)):
            if cluster[j] in  adj_list[cluster[i]]:
                internal_trading += adj_list[cluster[i]][cluster[j]]
        # external trading
        for j in trader_ids:
            if j in cluster_set:
                continue
            
            # i is seller and j is buyer
            if j in adj_list[cluster[i]]:
                external_trading += adj_list[cluster[i]][j]
                
            # j is seller and i is buyer
            if i in adj_list[j]:
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


# k,m,h compatiblity
def check_compatibility(cluster_1, cluster_2):
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
    print(f'Collusion levels size: {len(collusion_levels)}')
    compatible_pair_exists = False
    for i in range(len(collusion_levels)):
        #print(i, end=' ')
        cluster_1 = collusion_levels[i][1]
        cluster_2 = collusion_levels[i][2]
        collusion_level = collusion_levels[i][0]
        if collusion_level <= 0:
            continue
        if (check_compatibility(cluster_1, cluster_2) and
            check_compatibility(cluster_2, cluster_1)):
            collusion_sets.remove(cluster_1)
            collusion_sets.remove(cluster_2)
            collusion_sets.append(cluster_1 + cluster_2)
            compatible_pair_exists = True
            break
    if not compatible_pair_exists:
        break