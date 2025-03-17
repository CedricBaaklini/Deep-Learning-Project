import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)

edge_attr = torch.tensor([[0.96], [0.96], [0.96], [0.96]], dtype=torch.float)

data = Data(edge_index=edge_index, edge_attr=edge_attr)

print("edge_index:\n", data.edge_index)
print("edge_attr:\n", data.edge_attr)

print(data.keys())

print(data['x'])

for key, item in data:
    print(f'{key} found in data')

print('edge_attr' in data)

print(data.num_nodes)

print(data.num_edges)

print(data.num_node_features)

print(data.has_isolated_nodes())

print(data.has_self_loops())

print(data.is_directed())

device = torch.device('cuda')

data = data.to(device)