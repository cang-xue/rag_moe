from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


DEFAULT_CITIES = [
    "Delivery_SH",
    "Delivery_HZ",
    "Delivery_CQ",
    "Delivery_YT",
    "Delivery_JL",
]


@dataclass(frozen=True)
class TrainingProtocol:
    source_cities: List[str]
    target_city: str


def parse_city_list(raw) -> List[str]:
    if isinstance(raw, str):
        cities = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        cities = [str(item).strip() for item in raw if str(item).strip()]
    if not cities:
        raise ValueError("city list must not be empty")
    seen = set()
    duplicates = []
    for city in cities:
        if city in seen:
            duplicates.append(city)
        seen.add(city)
    if duplicates:
        raise ValueError("duplicate cities are not allowed: %s" % ",".join(sorted(set(duplicates))))
    return cities


def split_source_target(cities: Sequence[str], target_city: str) -> Tuple[List[str], str]:
    city_list = parse_city_list(cities)
    if target_city not in city_list:
        raise ValueError("target_city %r is not in city list %r" % (target_city, city_list))
    source_cities = [city for city in city_list if city != target_city]
    if not source_cities:
        raise ValueError("at least one source city is required")
    return source_cities, target_city


def build_protocol(cities: Iterable[str], target_city: str) -> TrainingProtocol:
    source_cities, target = split_source_target(list(cities), target_city)
    return TrainingProtocol(source_cities=source_cities, target_city=target)

