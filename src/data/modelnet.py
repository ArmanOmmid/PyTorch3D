import os
import numpy as np

import open3d as o3d
from torch.utils.data import Dataset

class ModelNet(Dataset):

    def __init__(self, path, samples, train=True) -> None:
        super().__init__()

        self.path = path
        self.split = "train" if train else "test"
        self.samples = samples

        self.inputs = []
        self.targets = []
        self.classes = []
        self.transform = None

        # Iterate through every item in the data directory
        for dirname in os.listdir(path):
            class_dir = f"{path}/{dirname}"
            # Skip over non-directories
            if not os.path.isdir(class_dir):
                continue

            # Class index is induced from the current length of the classes
            class_index = len(self.classes)
            # Name of directory is the name of the class
            self.classes.append(dirname)

            # Iterate through every file in the corresponding split
            class_split_dir = f"{class_dir}/{self.split}"
            for filename in os.listdir(class_split_dir):

                filepath = f"{class_split_dir}/{filename}"

                self.inputs.append(filepath)
                self.targets.append(class_index)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
            
            input = o3d.io.read_triangle_mesh(self.inputs[index])
            target = self.targets[index]

            # mesh = mesh.simplify_vertex_clustering(voxel_size=0.01, contraction=o3d.geometry.SimplificationContraction.Average)
            input = input.sample_points_uniformly(number_of_points=self.samples)
            input = np.asarray(input.points)

            if self.transform is not None:
                input = self.transform(input)

            return input, target
    
    def raw(self, index):
         return self.inputs[index], self.targets[index]
            


            
