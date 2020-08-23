import mmcv
import numpy as np

from .builder import DATASETS


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassSampledDataset(object):
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, ann_file):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self._ori_len = len(self.dataset)
        self.dataset.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        _cls_infos = {name: [] for name in self.CLASSES}
        for info in data['infos']:
            if self.dataset.use_valid_flag:
                mask = info['valid_flag']
                gt_names = set(info['gt_names'][mask])
            else:
                gt_names = set(info['gt_names'])
            for name in gt_names:
                if name in self.CLASSES:
                    _cls_infos[name].append(info)
        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {
            k: len(v) / duplicated_samples
            for k, v in _cls_infos.items()
        }

        data_infos = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in _cls_dist.values()]
        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
            data_infos += np.random.choice(cls_infos,
                                           int(len(cls_infos) *
                                               ratio)).tolist()

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        return self.dataset[idx]
