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
        return ["new_combined.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        import ast, pandas as pd, torch
        from torch_geometric.data import Data

        df = pd.read_csv(self.raw_paths[0])
        df = df.dropna(subset=["Area AVG", "RG AVG", "RDF Peak"])
        df = df[df["RDF Peak"].ne(0)]

        data_list = []
        for _, row in df.iterrows():
            input_list = ast.literal_eval(row["Input List"])

            features = []
            backbone_indices = {}
            for idx, (n, label) in enumerate(input_list):
                features.append([1, 0, 0])
                backbone_indices[n] = idx

            edge_index = [[], []]
            for i in range(len(input_list) - 1):
                edge_index[0].extend([i, i + 1])
                edge_index[1].extend([i + 1, i])

            next_node = len(features)
            for n, label in input_list:
                if label not in ("E0", "S0"):
                    m = int(label[1:])
                    bead_type = label[0]
                    prev = backbone_indices[n]
                    for _ in range(m):
                        features.append([0, 1, 0] if bead_type == "S" else [0, 0, 1])
                        curr = next_node
                        edge_index[0].extend([prev, curr])
                        edge_index[1].extend([curr, prev])
                        prev = curr
                        next_node += 1

            x = torch.tensor(features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            y = torch.tensor([row["Area AVG"], row["RG AVG"], row["RDF Peak"]],
                             dtype=torch.float32)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == "__main__":
    dataset = PolymerDataset(root=".")
    print(dataset)