"""Connector registry — maps a source_id to its Connector class.

`make ingest SOURCE=<id>` looks the id up here. Add a connector, register it, done.
"""

from __future__ import annotations

from ..base import Connector
from .demo import DemoConnector
from .openbrewerydb import OpenBreweryDBConnector
from .openfoodfacts import OpenFoodFactsConnector
from .ttb_cola import TTBColaConnector

REGISTRY: dict[str, type[Connector]] = {
    DemoConnector.source_id: DemoConnector,
    OpenBreweryDBConnector.source_id: OpenBreweryDBConnector,
    OpenFoodFactsConnector.source_id: OpenFoodFactsConnector,
    TTBColaConnector.source_id: TTBColaConnector,
}

# Friendly aliases so the Makefile can say SOURCE=ttb_cola
ALIASES = {
    "demo": "bcd-demo",
    "bcd-demo": "bcd-demo",
    "openbrewerydb": "openbrewerydb",
    "obdb": "openbrewerydb",
    "openfoodfacts": "openfoodfacts",
    "off": "openfoodfacts",
    "ttb_cola": "ttb-cola-registry",
    "ttb": "ttb-cola-registry",
}


def get_connector(store, source: str) -> Connector:
    source_id = ALIASES.get(source, source)
    if source_id not in REGISTRY:
        raise KeyError(
            f"no connector for '{source}'. known: {sorted(REGISTRY)} "
            f"(aliases: {sorted(ALIASES)})"
        )
    return REGISTRY[source_id](store)


__all__ = ["REGISTRY", "ALIASES", "get_connector"]
