import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

from experiments.training.protocol import build_protocol


@dataclass(frozen=True)
class CityDataContext:
    city: str
    city_id: int
    loaders: Dict[str, object]
    llm_encoding: torch.Tensor
    num_nodes: int
    null_value: float
    scalers: object
    supports: list
    unknown_set: set
    known_set: set
    num_masked_nodes: int


def compute_node_sets(num_nodes: int, num_unknown_nodes: int, seed: int = 42):
    unknown_count = min(int(num_unknown_nodes), int(num_nodes))
    rng = np.random.RandomState(int(seed))
    unknown_set = set(rng.choice(list(range(int(num_nodes))), unknown_count, replace=False).tolist())
    known_set = set(range(int(num_nodes))) - unknown_set
    return unknown_set, known_set


def _default_llm_loader(city: str):
    return np.load("./data/llmvec_llama3_%s.npy" % city).astype("float32")


def _default_support_loader(city: str):
    from src.utils.graph_algo import load_graph_data

    _, _, adj_mat = load_graph_data("data/sensor_graph/adj_mx_%s.pkl" % city.lower())
    return adj_mat


def _as_supports(support_value):
    if isinstance(support_value, (list, tuple)):
        values = support_value
    else:
        values = [support_value]
    return [torch.as_tensor(value, dtype=torch.float32) for value in values]


def build_single_city_context(
    city: str,
    city_id: int,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    num_unknown_nodes: int = 10,
    num_masked_nodes: int = 6,
    seed: int = 42,
    dataloader_fn: Optional[Callable] = None,
    num_nodes_fn: Optional[Callable] = None,
    null_value_fn: Optional[Callable] = None,
    llm_loader_fn: Optional[Callable] = None,
    support_loader_fn: Optional[Callable] = None,
):
    from src.utils.helper import get_dataloader, get_null_value, get_num_nodes

    dataloader_fn = dataloader_fn or get_dataloader
    num_nodes_fn = num_nodes_fn or get_num_nodes
    null_value_fn = null_value_fn or get_null_value
    llm_loader_fn = llm_loader_fn or _default_llm_loader
    support_loader_fn = support_loader_fn or _default_support_loader

    loaders = dataloader_fn(
        os.path.join("./data", city).replace("\\", "/"),
        batch_size,
        input_dim,
        output_dim,
        include_metadata=True,
    )
    num_nodes = int(num_nodes_fn(city))
    unknown_set, known_set = compute_node_sets(num_nodes, num_unknown_nodes, seed=seed)
    llm_encoding = torch.as_tensor(llm_loader_fn(city), dtype=torch.float32)
    supports = _as_supports(support_loader_fn(city))
    scalers = loaders.get("scalers", [])
    return CityDataContext(
        city=city,
        city_id=int(city_id),
        loaders=loaders,
        llm_encoding=llm_encoding,
        num_nodes=num_nodes,
        null_value=float(null_value_fn(city)),
        scalers=scalers,
        supports=supports,
        unknown_set=unknown_set,
        known_set=known_set,
        num_masked_nodes=int(num_masked_nodes),
    )


def build_city_contexts(
    cities: Iterable[str],
    target_city: str,
    batch_size,
    input_dim,
    output_dim,
    num_unknown_nodes: int = 10,
    num_masked_nodes: int = 6,
    seed: int = 42,
    dataloader_fn: Optional[Callable] = None,
    num_nodes_fn: Optional[Callable] = None,
    null_value_fn: Optional[Callable] = None,
    llm_loader_fn: Optional[Callable] = None,
    support_loader_fn: Optional[Callable] = None,
):
    protocol = build_protocol(cities, target_city)
    contexts = {}
    for city_id, city in enumerate(protocol.source_cities):
        contexts[city] = build_single_city_context(
            city=city,
            city_id=city_id,
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            num_unknown_nodes=num_unknown_nodes,
            num_masked_nodes=num_masked_nodes,
            seed=seed,
            dataloader_fn=dataloader_fn,
            num_nodes_fn=num_nodes_fn,
            null_value_fn=null_value_fn,
            llm_loader_fn=llm_loader_fn,
            support_loader_fn=support_loader_fn,
        )
    return protocol, contexts


def build_target_city_context(
    cities: Iterable[str],
    target_city: str,
    batch_size,
    input_dim,
    output_dim,
    num_unknown_nodes: int = 10,
    num_masked_nodes: int = 6,
    seed: int = 42,
    dataloader_fn: Optional[Callable] = None,
    num_nodes_fn: Optional[Callable] = None,
    null_value_fn: Optional[Callable] = None,
    llm_loader_fn: Optional[Callable] = None,
    support_loader_fn: Optional[Callable] = None,
):
    protocol = build_protocol(cities, target_city)
    return build_single_city_context(
        city=protocol.target_city,
        city_id=-1,
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim,
        num_unknown_nodes=num_unknown_nodes,
        num_masked_nodes=num_masked_nodes,
        seed=seed,
        dataloader_fn=dataloader_fn,
        num_nodes_fn=num_nodes_fn,
        null_value_fn=null_value_fn,
        llm_loader_fn=llm_loader_fn,
        support_loader_fn=support_loader_fn,
    )
