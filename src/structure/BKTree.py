from src.structure.custom_distance import CustomDistance

custom_distance = CustomDistance(threshold=0.5)


class BKTreeNode:
    def __init__(self, state, cluster_id):
        self.state = state
        self.cluster_id = cluster_id
        self.children = {}

    def add_child(self, dist, node):
        self.children[dist] = node


class BKTree:
    def __init__(self):
        self.root = None

    def find_node_by_cluster_id(self, cluster_id):
        """Recursively find the BKTreeNode with the requested cluster_id."""

        def search_node(node):
            if node.cluster_id == cluster_id:
                return node
            for child in node.children.values():
                result = search_node(child)
                if result:
                    return result
            return None

        if self.root:
            return search_node(self.root)
        return None


def find_max_cluster_id(node, max_cluster_id):
    """Recursively find the maximum cluster_id in the BK-Tree."""
    if node.cluster_id > max_cluster_id[0]:
        max_cluster_id[0] = node.cluster_id

    for child in node.children.values():
        find_max_cluster_id(child, max_cluster_id)


def get_max_cluster_id(bk_tree):
    """Return the maximum cluster_id in the BK-Tree."""
    max_cluster_id = [0]  # Use a mutable list so recursive calls can update the maximum.
    if bk_tree.root:
        find_max_cluster_id(bk_tree.root, max_cluster_id)
    return max_cluster_id[0]
