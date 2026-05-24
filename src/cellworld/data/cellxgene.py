from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.request import Request, urlopen


CELLXGENE_BASE_URL = "https://api.cellxgene.cziscience.com/curation/v1"


@dataclass(frozen=True)
class CellxGeneCollection:
    collection_id: str
    url: str
    name: str
    consortia: List[str]


def list_cellxgene_collections(limit: int = 20, timeout: float = 30.0) -> List[CellxGeneCollection]:
    """Fetch compact CELLxGENE collection metadata.

    The training code does not silently download large matrices. This function only returns
    metadata so a real-data experiment can choose datasets deliberately.
    """

    payload = _get_json(f"{CELLXGENE_BASE_URL}/collections", timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError("CELLxGENE collections endpoint returned an unexpected payload.")
    collections = []
    for item in payload[:limit]:
        collections.append(
            CellxGeneCollection(
                collection_id=str(item.get("collection_id", "")),
                url=str(item.get("collection_url", "")),
                name=str(item.get("name", "")),
                consortia=[str(value) for value in item.get("consortia", [])],
            )
        )
    return collections


def get_cellxgene_collection(collection_id: str, timeout: float = 30.0) -> Dict[str, Any]:
    return _get_json(f"{CELLXGENE_BASE_URL}/collections/{collection_id}", timeout=timeout)


def _get_json(url: str, timeout: float) -> Any:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "cellworld/0.1"})
    with urlopen(request, timeout=timeout) as response:  # nosec: public metadata endpoint
        return json.loads(response.read().decode("utf-8"))
