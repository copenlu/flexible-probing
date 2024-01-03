from typing import List, Optional, Dict, Tuple, Mapping
import torch
import copy
import warnings
from collections import Counter

from probekit.utils.types import Word, PyTorchDevice, PropertyValue


ClassificationDatasetDatastore = Dict[PropertyValue, List[Word]]


class DatasetTransform:
    def __init__(self):
        pass

    def transform(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, input: torch.Tensor):
        return self.transform(input)


class ClassificationDataset(Mapping[PropertyValue, List[Word]]):
    """
    A classification dataset amounts to an immutable dictionary, alongside some helper methods.

    See: https://stackoverflow.com/questions/21361106/how-would-i-implement-a-dict-with-abstract-base-classes-in-python  # noqa

    The `transform` is an optional argument that can be used to apply a transformation to every datapoint.
    """
    def __init__(self, data: ClassificationDatasetDatastore, device: PyTorchDevice = "cpu",
                 transform: Optional[DatasetTransform] = None):
        self._data = copy.deepcopy(data)
        self._device = device
        self._transform = transform

        # Generate tensorized version of the each properties embeddings
        self._embeddings_tensors = {
            property_value: torch.stack([w.get_embedding() for w in word_list], dim=0).to(self._device)
            for property_value, word_list in self._data.items()
        }

        # Generate concatenated versions of the inputs and the values
        self._embeddings_tensor_concat = torch.cat(
            [tensor for _, tensor in self._embeddings_tensors.items()], dim=0).to(self._device)
        self._values_tensor_concat = torch.cat(
            [torch.tensor([idx] * len(words))
             for idx, (_, words) in enumerate(self._data.items())], dim=0).to(self._device)

        if transform is not None:
            self._embeddings_tensor_concat = self._transform(self._embeddings_tensor_concat)

            # Update per-property tensor
            self._embeddings_tensors = {prop: self._transform(tensor) for prop, tensor in
                                        self._embeddings_tensors.items()}

    @staticmethod
    def get_property_value_list(attribute: str, *word_lists, min_count: int = 1) -> List[PropertyValue]:
        property_value_counters = [
            Counter([w.get_attribute(attribute) for w in word_list if w.has_attribute(attribute)])
            for word_list in word_lists]

        # 1. Build list of property values in all datasets
        property_value_sets = [set(pcv.keys()) for pcv in property_value_counters]
        kept_property_values = list(property_value_sets[0].intersection(*property_value_sets[1:]))

        # 2. Filter by counts
        final_property_values = []
        if min_count is not None:
            for kpv in kept_property_values:
                if all([pvc[kpv] >= min_count for pvc in property_value_counters]):
                    final_property_values.append(kpv)
        else:
            final_property_values = kept_property_values

        return sorted(final_property_values)

    @classmethod
    def from_word_list(cls, words: List[Word], attribute: str, device: PyTorchDevice = "cpu",
                       transform: Optional[DatasetTransform] = None,
                       property_value_list: Optional[List[PropertyValue]] = None):

        if property_value_list is None:
            warnings.warn("You have not specified a `property_value_list` to construct the "
                          "ClassificationDataset. This can lead to problems!")

        datastore: ClassificationDatasetDatastore = {}
        if property_value_list is not None:
            datastore = {prop: [] for prop in property_value_list}

        for w in words:
            if not w.has_attribute(attribute):
                continue

            property_value: PropertyValue = w.get_attribute(attribute)
            if property_value_list is None and property_value not in datastore:
                # We are constructing the dictionary completely from scratch, so add this key
                datastore[property_value] = []

            if property_value in datastore:
                datastore[property_value].append(w)

        # Ensure keys are in alphabetical order
        datastore = dict(sorted(datastore.items()))

        return cls(datastore, device=device, transform=transform)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        internal_string = ', '.join([f"{prop} [{len(words)}]" for prop, words in self._data.items()])
        return f"{type(self).__name__}({internal_string})"

    def get_dimensionality(self) -> int:
        return self._embeddings_tensor_concat.shape[1]

    def get_device(self) -> PyTorchDevice:
        return self._device

    def get_embeddings_tensors(
            self, select_dimensions: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        if select_dimensions:
            return {prop: tensor[:, select_dimensions] for prop, tensor in self._embeddings_tensors.items()}

        return self._embeddings_tensors

    def get_inputs_values_tensor(
            self, select_dimensions: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embeddings_tensor_concat
        if select_dimensions:
            embeddings = self._embeddings_tensor_concat[:, select_dimensions]

        return embeddings, self._values_tensor_concat


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Credits to: Jesse Mu (jayelm).
    See: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.device = tensors[0].device

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
