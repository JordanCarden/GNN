import ast
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

class PolymerDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def raw_file_names(self):
        return ['ES.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df = df.dropna(subset=['Area AVG', 'Area STD', 'RG AVG', 'RG STD', 'RDF Peak', 'Coordination at Minimum'])
        data_list = []
        for _, row in df.iterrows():
            input_list = ast.literal_eval(row['Input List'])
            features = []
            backbone_indices = {}
            for idx, (n, label) in enumerate(input_list):
                features.append([1,0,0])
                backbone_indices[n] = idx
            edge_index = [[], []]
            for i in range(len(input_list)-1):
                edge_index[0].append(i)
                edge_index[1].append(i+1)
                edge_index[0].append(i+1)
                edge_index[1].append(i)
            next_node = len(features)
            for n, label in input_list:
                if label not in ('E0', 'S0'):
                    m = int(label[1:])
                    bead_type = label[0]
                    prev = backbone_indices[n]
                    for _ in range(m):
                        features.append([0,1,0] if bead_type == 'S' else [0,0,1])
                        curr = next_node
                        edge_index[0].append(prev)
                        edge_index[1].append(curr)
                        edge_index[0].append(curr)
                        edge_index[1].append(prev)
                        prev = curr
                        next_node += 1
            x = torch.tensor(features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            y = torch.tensor([row['Area AVG'], row['Area STD'], row['RG AVG'], row['RG STD'], row['RDF Peak'], row['Coordination at Minimum']], dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = PolymerDataset(root='.')
    print(dataset)