import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
random.seed = 42
import os 
from typing import Tuple
CLASS= {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
class PointSampler(object):
    def __init__(self, output_size: int) -> None:

        """ Class initializer

            Params
            --------
                output_size (int): Output size of the network

        """

        assert isinstance(output_size, int)
        self.output_size = output_size
        return

    def triangle_area(
            self,
            pt1: np.ndarray,
            pt2: np.ndarray,
            pt3: np.ndarray
    ) -> np.float64:

        """ Calculates the area of a triangle

            Params
            --------
                pt1 (numpy.ndarray): Coordinate values for the 1st point
                pt2 (numpy.ndarray): Coordinate values for the 2nd point
                pt3 (numpy.ndarray): Coordinate values for the 3rd point

            Returns
            --------
                ret (np.float64): Area of the triangle

        """

        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        ret = max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5
        return ret

    def sample_point(
            self,
            pt1: np.ndarray,
            pt2: np.ndarray,
            pt3: np.ndarray
    ) -> tuple:

        """ Samples points with barycentric coordinates on a triangle

            Reference: https://mathworld.wolfram.com/BarycentricCoordinates.html

            Params
            --------
                pt1 (numpy.ndarray): Coordinate values for the 1st point
                pt2 (numpy.ndarray): Coordinate values for the 2nd point
                pt3 (numpy.ndarray): Coordinate values for the 3rd point

            Returns
            --------
                ret (np.float64): Area of the triangle

        """

        s, t = sorted([random.random(), random.random()])
        coords = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return coords(0), coords(1), coords(2)

    def __call__(self, mesh: tuple) -> np.ndarray:

        """ Class caller

            Params
            --------
                mesh (tuple): Mesh data

            Returns
            --------
                sampled_points (numpy.ndarray): Sample point data

        """

        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (
                self.triangle_area(
                    verts[faces[i][0]],
                    verts[faces[i][1]],
                    verts[faces[i][2]]
                )
            )

        sampled_faces = (
            random.choices(
                faces,
                weights=areas,
                cum_weights=None,
                k=self.output_size
            )
        )

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (
                self.sample_point(
                    verts[sampled_faces[i][0]],
                    verts[sampled_faces[i][1]],
                    verts[sampled_faces[i][2]]
                )
            )

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:

        """ Class caller

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data

            Returns
            --------
                norm_pointcloud (numpy.ndarray): Normalized point cloud data

        """

        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud
class ToTensor(object):
    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:

        """ Converts data type of pointcloud from numpy array to torch tensor

            Params
            --------
                pointcloud (numpy.ndarray): Point cloud data array

            Returns
            --------
                torch.Tensor

        """

        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)
def default_transforms() -> transforms.Compose:

    """ Returns data transforms

        Returns
        --------
            torchvision.transforms.transforms.Compose

    """

    return transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            ToTensor()
        ]
    )
def read_off(file) -> Tuple[list, list]:

    """ Reads 'OFF' headers

        Params
        --------
            file (_io.TextIOWrapper): file IO

        Returns
        --------
            verts (list): List of vertices
            faces (list): List of faces

    """

    if 'OFF' != file.readline().strip():
        raise 'Not a valid OFF header'

        
    n_verts, n_faces, _ = tuple([
        int(s) for s in file.readline().strip().split(' ')
    ])
    verts = [
        [float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)
    ]
    faces = [
        [int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)
    ]
    
    return verts, faces

class PointCloudData(Dataset):
    def __init__(
        self,
        root_dir: str,
        folder: str = "train",
        transform: transforms.Compose = default_transforms()
    ) -> None:

        """ Class initializer

            Params
            --------
                root_dir (str): Path to data root
                folder (str): Name of 'train' folder
                transform (torchvision.transforms.transforms.Compose): Data transform

        """

        self.root_dir = root_dir
        folders = [
            _dir for _dir in sorted(os.listdir(root_dir))\
            if os.path.isdir(os.path.join(root_dir, _dir))
        ]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []

        for category in self.classes.keys():
            new_dir = os.path.join(root_dir, category, folder)
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = dict()
                    sample['pcd_path'] = os.path.join(new_dir, file)
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self) -> int:

        """ Returns the length of files
        """

        return len(self.files)

    def __preproc__(self, file) -> torch.Tensor:

        """ Calculates the transformation of the pointcloud data

            Params
            --------
                file (_io.TextIOWrapper): File IO

            Returns
            --------
                pointcloud (torch.Tensor): Transformed point cloud data

        """

        verts, faces = read_off(file)
        
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx: int):

        """ Returns pointcloud data and category

            Params
            --------
                idx (int): Index of data

            Returns
            --------
                ret (dict): Pointcloud and its category

        """

        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)

        ret = {
            'pointcloud': pointcloud,
            'category': self.classes[category]
        }
        return ret
def get_dataset(
    data_path: str,
    folder: str,
    dataset_type: str = 'train'
) -> PointCloudData:

    """ Gets dataset

            Params
            --------
                data_path (str):
                folder (str):
                dataset_type (str):

            Returns
            --------
                ds (PointCloudData): PointCloudData class (dataset)

        """

    ds = None
    if dataset_type == 'train':
        train_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                RandRotation_z(),
                RandomNoise(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=train_transforms
        )
    elif dataset_type == 'valid':
        valid_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=valid_transforms
        )
    elif dataset_type == 'test':
        test_transforms = transforms.Compose(
            [
                PointSampler(1024),
                Normalize(),
                ToTensor()
            ]
        )
        ds = PointCloudData(
            root_dir=data_path,
            folder=folder,
            transform=test_transforms
        )
    else:
        raise ValueError('dataset type mismatches.')

    inv_classes = {i: cat for cat, i in ds.classes.items()};
    print(inv_classes)

    print(f'\nDataset size for {dataset_type}: ', len(ds))
    print('Number of classes: ', len(ds.classes))

    return ds
def get_dataloader(
    data_path: str,
    folder: str,
    dataset_type: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True
) -> DataLoader:

    """ Gets dataloader

        Params
        --------
            data_path (str): Path to data root
            folder (str): Name of 'train' folder
            dataset_type (str): Type of dataset (train/valid/test)
            batch_size (int): Batch size
            num_workers (int): Number of workers for data pipeline
            pin_memory (bool): Use pin memory?
            shuffle (bool): Shuffle data?

        Returns
        --------
            data_loader (torch.utils.data.dataloader.Dataloader): Dataloader

    """

    dataset = get_dataset(
        data_path=data_path,
        folder=folder,
        dataset_type=dataset_type
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return data_loader