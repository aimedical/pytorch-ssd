import numpy as np
import cv2
import os
from PIL import Image
from torchvision.datasets import VisionDataset


class CocoDetection(VisionDataset):
    # def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = ['BACKGROUND']
        for c in cats:
            self.class_names.append(c['name'])
        self.class_names = tuple(self.class_names)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _read_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for item in ann:
            b = item['bbox']
            boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            labels.append(item['category_id'] + 1)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        image = self._read_image(os.path.join(self.root, path))
        boxes, labels = self._get_annotation(img_id)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def __len__(self):
        return len(self.ids)
