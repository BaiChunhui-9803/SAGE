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
        """
        递归查找指定 cluster_id 的 BKTreeNode
        """

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
    """
    递归查找 BKTree 中最大的 cluster_id
    """
    if node.cluster_id > max_cluster_id[0]:
        max_cluster_id[0] = node.cluster_id

    for child in node.children.values():
        find_max_cluster_id(child, max_cluster_id)


def get_max_cluster_id(bk_tree):
    """
    获取 BKTree 中最大的 cluster_id
    """
    max_cluster_id = [0]  # 使用列表来存储最大值，以便在递归中修改
    if bk_tree.root:
        find_max_cluster_id(bk_tree.root, max_cluster_id)
    return max_cluster_id[0]
