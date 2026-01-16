import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset


def preprocess_mask(img):
    mask = np.zeros_like(img)
    mask[img >= 150] = 1
    return mask


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=2, shift_segmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l)
                                   for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        segmentation = Image.open(example["segmentation_path_"])
        if not segmentation.mode == "L":
            segmentation = segmentation.convert("L")
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)

        # Preprocess
        segmentation = preprocess_mask(segmentation)

        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }
        # print("222:",image.shape)
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        # 假设 onehot 是你的 onehot tensor
        # print("one_hot:",onehot.shape)[256,256,2]
        onehot_permuted = np.transpose(onehot, (2,0,1))
        # print("one_hot_end:",onehot_permuted.shape)[2,256,256]
        # print("================================================================================S")
        example["segmentation"] = onehot_permuted.astype(np.float32)
        return example


class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/sflckr_examples.txt",
                         data_root="data/sflckr_images",
                         segmentation_root="data/sflckr_segmentations",
                         size=size, random_crop=random_crop, interpolation=interpolation)


class KvasirSegTrain(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='/home/hyl/yujia/data_5000_384/train.txt',
                         data_root='/home/hyl/yujia/data_5000_384/images_256',
                         segmentation_root='/home/hyl/yujia/data_5000_384/masks_grey_256',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)
class Fake_mask(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='/home/hyl/yujia/experiments/data/data_256/TrainDataset/train.txt',
                         data_root='/home/hyl/yujia/experiments/data/data_256/TrainDataset/images',
                         segmentation_root='/home/hyl/yujia/experiments/data/data_256/TrainDataset/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)

class KvasirSegEval(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='/home/hyl/yujia/test_data/TestData_verge/train.txt',
                         data_root='/home/hyl/yujia/test_data/TestData_verge/images',
                         segmentation_root='/home/hyl/yujia/test_data/TestData_verge/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)
class KvasirSegTest(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='data/kvasir/kvasir_eval.txt',
                         data_root='data/kvasir/images',
                         segmentation_root='data/kvasir/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)
class uncondition(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='data/kvasir/kvasir_eval.txt',
                         data_root='data/kvasir/images',
                         segmentation_root='data/kvasir/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)


def write_lines(file, lines):
    with open(file, 'w') as f:
        for line in lines:
            f.write(os.path.basename(line))
            f.write('\n')


def generateKvasirCSV(dir, output, train=0.9):
    files = glob.glob(dir)
    random.shuffle(files)
    length = len(files)

    train_data = files[:int(train * length)]
    write_lines(f'{output}/kvasir_train.txt', train_data)

    eval_data = files[int(train * length):]
    write_lines(f'{output}/kvasir_eval.txt', eval_data)
