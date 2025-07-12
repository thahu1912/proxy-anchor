from .base import *
import os

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = os.path.join(root, 'online_products')
        self.mode = mode
        self.transform = transform
        info_file = None
        if self.mode == 'train':
            self.classes = range(0,11318)
            info_file = 'Ebay_train.txt'
        elif self.mode == 'eval':
            self.classes = range(11318,22634)
            info_file = 'Ebay_test.txt'
        if info_file is None:
            raise ValueError(f"Unknown mode: {self.mode}")

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata_path = os.path.join(self.root, 'Info_Files', info_file)
        with open(metadata_path) as metadata:
            for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
                if i > 0:
                    if int(class_id)-1 in self.classes:
                        self.ys += [int(class_id)-1]
                        self.I += [int(image_id)-1]
                        self.im_paths.append(os.path.join(self.root, 'images', path))