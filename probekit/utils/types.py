from typing import List, Union, Dict, Any
import torch


class Word:
    def __init__(self, word: str, embedding: torch.Tensor, attributes: Dict[str, str]) -> None:
        self._word = word
        self._embedding = embedding
        self._attributes = attributes

    def get_word(self) -> str:
        return self._word

    def get_embedding(self) -> torch.Tensor:
        return self._embedding

    def has_attribute(self, attr) -> bool:
        return attr in self._attributes

    def get_attribute(self, attr) -> Any:
        return self._attributes[attr]

    def get_attributes(self) -> List[str]:
        return list(self._attributes.keys())

    def __repr__(self) -> str:
        return "{}({})".format(self._word, self._attributes)


PyTorchDevice = Union[torch.device, str]
Specification = Dict[str, Any]
PropertyValue = str
