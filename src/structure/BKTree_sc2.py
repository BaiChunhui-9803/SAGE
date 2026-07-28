class ClusterNode:
    def __init__(self, state, cluster_id):
        self.state = state
        self.cluster_id = cluster_id
        self.children = {}
        self.state_list = []

    def add_child(self, distance, child):
        self.children[distance] = child

    def add_state(self, state):
        self.state_list.append(state)


class BKTree:
    def __init__(self, distance_func, distance_index=0):
        self.root = None
        self.distance_func = distance_func
        self.distance_index = distance_index
        self.next_cluster_id = 2

    def insert(self, node, parent=None):
        if parent is None:
            self.root = node
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            return
        dist = self.distance_func(node.state, parent.state)[self.distance_index]
        if dist in parent.children:
            self.insert(node, parent.children[dist])
        else:
            parent.add_child(dist, node)
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1

    def query(self, state, threshold):
        def search(node, dist):
            if dist < threshold:
                return node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < threshold:
                    result = search(
                        child,
                        self.distance_func(state, child.state)[self.distance_index],
                    )
                    if result is not None:
                        return result
            return None

        if self.root is None:
            return None
        return search(
            self.root, self.distance_func(state, self.root.state)[self.distance_index]
        )

    def query_nearest(self, state):
        best_id = None
        best_dist = float("inf")

        def search(node, dist):
            nonlocal best_id, best_dist
            if dist < best_dist:
                best_dist = dist
                best_id = node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < best_dist:
                    search(
                        child,
                        self.distance_func(state, child.state)[self.distance_index],
                    )

        if self.root is None:
            return None, float("inf")
        search(
            self.root, self.distance_func(state, self.root.state)[self.distance_index]
        )
        return best_id, best_dist

    def find_node_by_cluster_id(self, cluster_id):
        if self.root is None:
            return None

        def search_node(node):
            if node.cluster_id == cluster_id:
                return node
            for child in node.children.values():
                result = search_node(child)
                if result is not None:
                    return result
            return None

        return search_node(self.root)

    def get_next_cluster_id(self):
        return self.next_cluster_id


def classify_new_state(new_state, bktree, threshold=1.0):
    cluster_id = bktree.query(new_state, threshold)
    if cluster_id is not None:
        return cluster_id
    else:
        new_cluster_id = bktree.get_next_cluster_id()
        new_node = ClusterNode(new_state, new_cluster_id)
        bktree.insert(new_node, bktree.root)
        return new_cluster_id


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
