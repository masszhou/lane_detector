from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.augmentation import Flip, Translate, Rotate, AddGaussianNoise, ChangeIntensity
from dataset.transform_op import Resize, TransposeNumpyArray, NormalizeInstensity

from dataset.tusimple import DatasetTusimple
from dataset.bdd100k import DatasetBDD100K
from dataset.culane import DatasetCULane
from dataset.collections import DatasetCollections


def build_dataset(params):
    if params.dataset_name == "tusimple":
        return get_tusimple(params)
    elif params.dataset_name == "bdd100k":
        return get_bdd100k(params)
    elif params.dataset_name == "culane":
        return get_culane(params)
    elif params.dataset_name == "collections":
        return get_collections(params)
    else:
        return None, None


def build_dataloader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=dataset.collate_fn)
    return data_loader


def get_tusimple(params):
    # augmentation
    flip = Flip()
    translate = Translate()
    rotate = Rotate()
    add_noise = AddGaussianNoise()
    change_intensity = ChangeIntensity()
    resize = Resize(rows=256, cols=512)
    norm_to_1 = NormalizeInstensity()
    whc_to_cwh = TransposeNumpyArray((2, 0, 1))

    train_dataset = DatasetTusimple(root_path=params.train_root_url,
                                    json_files=params.train_json_file,
                                    transform=transforms.Compose([flip,
                                                                  translate,
                                                                  rotate,
                                                                  add_noise,
                                                                  change_intensity,
                                                                  resize,
                                                                  norm_to_1,
                                                                  whc_to_cwh]), )
    val_dataset = DatasetTusimple(params.val_root_url,
                                  params.val_json_file,
                                  transform=transforms.Compose([resize,
                                                                norm_to_1,
                                                                whc_to_cwh]), )
    return train_dataset, val_dataset


def get_bdd100k(params):
    # augmentation
    flip = Flip()
    translate = Translate()
    rotate = Rotate()
    add_noise = AddGaussianNoise()
    change_intensity = ChangeIntensity()
    resize = Resize(rows=256, cols=512)
    norm_to_1 = NormalizeInstensity()
    whc_to_cwh = TransposeNumpyArray((2, 0, 1))

    train_dataset = DatasetBDD100K(root_path=params.train_root_url,
                                   json_files=params.train_json_file,
                                   transform=transforms.Compose([flip,
                                                                 translate,
                                                                 rotate,
                                                                 add_noise,
                                                                 change_intensity,
                                                                 resize,
                                                                 norm_to_1,
                                                                 whc_to_cwh]), )

    return train_dataset, None


def get_culane(params):
    # augmentation
    flip = Flip()
    translate = Translate()
    rotate = Rotate()
    add_noise = AddGaussianNoise()
    change_intensity = ChangeIntensity()
    resize = Resize(rows=256, cols=512)
    norm_to_1 = NormalizeInstensity()
    whc_to_cwh = TransposeNumpyArray((2, 0, 1))

    train_dataset = DatasetCULane(root_path=params.train_root_url,
                                  index_file=params.train_json_file,
                                  transform=transforms.Compose([flip,
                                                                translate,
                                                                rotate,
                                                                add_noise,
                                                                change_intensity,
                                                                resize,
                                                                norm_to_1,
                                                                whc_to_cwh]), )

    return train_dataset, None


def get_collections(params):
    # augmentation
    flip = Flip()
    translate = Translate()
    rotate = Rotate()
    add_noise = AddGaussianNoise()
    change_intensity = ChangeIntensity()
    resize = Resize(rows=256, cols=512)
    norm_to_1 = NormalizeInstensity()
    whc_to_cwh = TransposeNumpyArray((2, 0, 1))

    train_dataset = DatasetCollections(transform=transforms.Compose([flip,
                                                                     translate,
                                                                     rotate,
                                                                     add_noise,
                                                                     change_intensity,
                                                                     resize,
                                                                     norm_to_1,
                                                                     whc_to_cwh]), )
    val_dataset = DatasetTusimple(params.val_root_url,
                                  params.val_json_file,
                                  transform=transforms.Compose([resize,
                                                                norm_to_1,
                                                                whc_to_cwh]), )
    return train_dataset, val_dataset