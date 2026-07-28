"""
BK-Tree structure module
Migrated from structure/ClusterCentricBKTree/
"""

import json
from typing import Dict, List, Any, Optional, Callable

from .state_distance import custom_distance


class BKTreeNode:
    def __init__(self, state: Dict, cluster_id: int):
        self.state = state
        self.cluster_id = cluster_id
        self.children: Dict[float, "BKTreeNode"] = {}

    def add_child(self, distance: float, child_node: "BKTreeNode"):
        self.children[distance] = child_node

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "cluster_id": self.cluster_id,
            "children": {str(k): v.to_dict() for k, v in self.children.items()},
        }


class BKTree:
    def __init__(self):
        self.root: Optional[BKTreeNode] = None

    def insert(self, state: Dict, cluster_id: int, distance_func: Callable):
        if self.root is None:
            self.root = BKTreeNode(state, cluster_id)
            return

        node = BKTreeNode(state, cluster_id)
        current = self.root

        while True:
            dist = distance_func(state, current.state)
            if dist in current.children:
                current = current.children[dist]
            else:
                current.add_child(dist, node)
                break

    def search(
        self, query: Dict, max_distance: float, distance_func: Callable
    ) -> List[tuple]:
        results = []
        if self.root is None:
            return results

        candidates = [self.root]
        while candidates:
            current = candidates.pop()
            dist = distance_func(query, current.state)

            if dist <= max_distance:
                results.append((current, dist))

            for d, child in current.children.items():
                if dist - max_distance <= d <= dist + max_distance:
                    candidates.append(child)

        return results

    def save(self, filepath: str):
        if self.root is None:
            return
        with open(filepath, "w") as f:
            json.dump(self.root.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BKTree":
        tree = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        tree.root = cls._deserialize_node(data)
        return tree

    @staticmethod
    def _deserialize_node(data: Dict) -> BKTreeNode:
        node = BKTreeNode(data["state"], data["cluster_id"])
        for dist_str, child_data in data["children"].items():
            child_node = BKTree._deserialize_node(child_data)
            node.add_child(float(dist_str), child_node)
        return node


def get_max_cluster_id(tree: BKTree) -> int:
    if tree.root is None:
        return 0

    max_id = tree.root.cluster_id
    candidates = [tree.root]

    while candidates:
        current = candidates.pop()
        max_id = max(max_id, current.cluster_id)
        for child in current.children.values():
            candidates.append(child)

    return max_id
