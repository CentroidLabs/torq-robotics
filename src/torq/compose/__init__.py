# torq.compose — dataset composition and query

from torq.compose._compose import compose
from torq.compose._query import query
from torq.compose.dataset import Dataset

__all__ = ["Dataset", "compose", "query"]
