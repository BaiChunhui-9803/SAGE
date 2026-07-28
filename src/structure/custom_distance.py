import numpy as np
from scipy.optimize import linear_sum_assignment

class DistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_coordinates_and_health(self, state, army_type):
        """
        从状态中提取特定军队类型的所有坐标和生命值信息。
        :param state: 状态字典
        :param army_type: 军队类型 ('blue_army' 或 'red_army')
        :return: 包含所有坐标的二维数组和生命值的一维数组
        """
        coordinates = []
        health_values = []
        for unit in state[army_type]:
            coordinates.append(unit[:2])  # 提取前两个元素作为坐标
            health_values.append(unit[2])  # 假设生命值是第三个元素
        return np.array(coordinates), np.array(health_values)

    def calculate_distance_and_health_difference(self):
        """
        计算两个状态之间的距离和生命值分布的差异。
        :return: tuple, (float, float)。第一个返回值是坐标分布的距离，第二个返回值是生命值分布的差异
        """
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_distance = 0.0
        total_health_difference = 0.0

        # 定义一个较大的值作为虚拟点的距离
        max_distance = 1.0

        # 处理 blue_army
        coords1_blue, health1_blue = self._extract_coordinates_and_health(self.state_1['state'][0], 'blue_army')
        coords2_blue, health2_blue = self._extract_coordinates_and_health(self.state_2['state'][0], 'blue_army')

        # 处理 red_army
        coords1_red, health1_red = self._extract_coordinates_and_health(self.state_1['state'][0], 'red_army')
        coords2_red, health2_red = self._extract_coordinates_and_health(self.state_2['state'][0], 'red_army')

        # 定义一个辅助函数来计算距离矩阵并进行匹配
        def calculate_army_distance_and_health_difference(coords1, coords2, health1, health2, max_distance):
            if len(coords1) == 0 and len(coords2) == 0:
                return 0.0, 0.0  # 如果两个数组都为空，返回 (0.0, 0.0)
            elif len(coords1) == 0 or len(coords2) == 0:
                # 如果其中一个为空，返回最大距离和生命值差异乘以非空的点数
                return max_distance * max(len(coords1), len(coords2)), 1.0  # 返回1.0作为生命值差异的乘积
            else:
                # 计算坐标距离矩阵
                distance_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)
                max_len = max(len(coords1), len(coords2))
                distance_matrix = np.pad(distance_matrix,
                                         ((0, max_len - len(coords1)), (0, max_len - len(coords2))),
                                         mode='constant', constant_values=max_distance)
                # 匹配最近点
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                # 计算匹配点的坐标距离
                total_distance = distance_matrix[row_ind, col_ind].sum()

                # 计算匹配点的生命值差异
                health_difference_product = 0.0
                for r, c in zip(row_ind, col_ind):
                    if r < len(health1) and c < len(health2):
                        health_difference_product += abs(health1[r] - health2[c])
                    elif r < len(health1):
                        health_difference_product += abs(health1[r])  # 与虚拟点的差异
                    elif c < len(health2):
                        health_difference_product += abs(health2[c])  # 与虚拟点的差异

                return total_distance, health_difference_product

        # 计算 blue_army 的距离和生命值差异
        distance_blue, health_difference_blue = calculate_army_distance_and_health_difference(
            coords1_blue, coords2_blue, health1_blue, health2_blue, max_distance
        )
        total_distance += distance_blue
        total_health_difference += health_difference_blue

        # 计算 red_army 的距离和生命值差异
        distance_red, health_difference_red = calculate_army_distance_and_health_difference(
            coords1_red, coords2_red, health1_red, health2_red, max_distance
        )
        total_distance += distance_red
        total_health_difference += health_difference_red

        return total_distance, total_health_difference

    def __call__(self):
        return self.calculate_distance_and_health_difference()

class CustomDistance:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def multi_distance(self, obs1, obs2):
        """
        判断两个状态是否在分布上相同，并计算生命值分布的差异。
        :param obs1: 第一个状态
        :param obs2: 第二个状态
        :return: tuple, (bool, float)。第一个返回值表示两个状态是否在分布上相同，第二个返回值表示生命值分布的差异
        """
        # 计算两个状态的距离和生命值差异
        distance_calculator = DistributionDistance(obs1, obs2)
        distribution_distance, health_distance = distance_calculator()  # 调用 DistributionDistance 的 __call__ 方法来获取距离值
        return distribution_distance, health_distance