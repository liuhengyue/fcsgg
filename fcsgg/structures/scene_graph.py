"""
The core class to store ground-truth, intermediate variables, and results related to object detection and scene graphs.

implementation modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/instances.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import itertools
from typing import Any, Dict, List, Tuple, Union
import torch


class SceneGraph:
    """
    This class represents a scene graph of an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields",
    additionally with relationships of instances.

    The differences from detectron2 Instances are that, here the length of each field is
    NOT enforced to be the same length.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``
    """
    _DEFAULT_FIELDS = ["gt_classes", "gt_ct_maps", "gt_wh", "gt_reg", "gt_centers_int",
                      "gt_relations", "gt_relations_weights", "gt_index", "gt_num_relations"]

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        self._extra_fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def init(self):
        for k in self._DEFAULT_FIELDS:
            self.set(k, [])
    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        self._fields[name] = value

    def set_extra(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        self._extra_fields[name] = value

    def get_extra(self, name: str) -> None:
        """
        Get the field named `name` from _extra_fields.
        """
        return self._extra_fields[name]

    def append(self, name: str, value: Any) -> None:
        """
        Append `value` to the field named `name`.
        The field has to be list
        """
        assert isinstance(self._fields[name], list)
        self._fields[name].append(value)

    def update(self, value: Dict) -> None:
        """
        add the key-value pairs to the specific fields by `append`.
        """
        for k, v in value.items():
            self.append(k, v)

    def set_at(self, idx: int, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        self._fields[name][idx] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "SceneGraph":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = SceneGraph(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if hasattr(item, "to"):
                        v[i] = item.to(*args, **kwargs)
            ret.set(k, v)
        # deal with original gt infos
        for k, v in self._extra_fields.items():
            if hasattr(v, "to"):
                ret.set_extra(k, v.to(*args, **kwargs))
        return ret


    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "SceneGraph":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("SceneGraph index out of range!")
            else:
                # we just need to have integer slice
                # item = slice(item, None, len(self))
                pass

        ret = SceneGraph(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        for k, v in self.__dict__.items():
            if k not in ["_image_size", "_fields"]:
                ret.__setattr__(k, v)
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty SceneGraph does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`SceneGraph` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["SceneGraph"]) -> "SceneGraph":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, SceneGraph) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = SceneGraph(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
