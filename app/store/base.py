from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class BaseStore(ABC):
    _instances: dict[type, "BaseStore"] = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    @abstractmethod
    def build_from_df(self, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def search(self, product_id: str, **kwargs) -> Union[dict, list[str]]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
