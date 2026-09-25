import numpy as np
from scipy.optimize import linear_sum_assignment

class DistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_coordinates_and_health(self, state, army_type):
        """Extract coordinate and health arrays for the requested army type (blue_army or red_army) from a state dictionary."""
        coordinates = []
        health_values = []
        for unit in state[army_type]:
            coordinates.append(unit[:2])  # The first two elements are coordinates.
            health_values.append(unit[2])  # The third element is health.
        return np.array(coordinates), np.array(health_values)

    def calculate_distance_and_health_difference(self):
        """Return coordinate-assignment distance and health difference between two states."""
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_distance = 0.0
        total_health_difference = 0.0

        # Use this penalty for unmatched dummy points.
        max_distance = 1.0

        # Extract blue-army units.
        coords1_blue, health1_blue = self._extract_coordinates_and_health(self.state_1, 'blue_army')
        coords2_blue, health2_blue = self._extract_coordinates_and_health(self.state_2, 'blue_army')

        # Extract red-army units.
        coords1_red, health1_red = self._extract_coordinates_and_health(self.state_1, 'red_army')
        coords2_red, health2_red = self._extract_coordinates_and_health(self.state_2, 'red_army')

        # Compute the distance matrix and match units.
        def calculate_army_distance_and_health_difference(coords1, coords2, health1, health2, max_distance):
            if len(coords1) == 0 and len(coords2) == 0:
                return 0.0, 0.0  # Two empty armies have zero distance.
            elif len(coords1) == 0 or len(coords2) == 0:
                # For one empty army, scale the unmatched penalties by the nonempty unit count.
                return max_distance * max(len(coords1), len(coords2)), 1.0  # Use a health penalty of 1.0 per unmatched unit.
            else:
                # Compute pairwise coordinate distances.
                distance_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)
                max_len = max(len(coords1), len(coords2))
                distance_matrix = np.pad(distance_matrix,
                                         ((0, max_len - len(coords1)), (0, max_len - len(coords2))),
                                         mode='constant', constant_values=max_distance)
                # Find the minimum-cost unit assignment.
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                # Sum coordinate distances for matched units.
                total_distance = distance_matrix[row_ind, col_ind].sum()

                # Sum health differences for matched units.
                health_difference_product = 0.0
                for r, c in zip(row_ind, col_ind):
                    if r < len(health1) and c < len(health2):
                        health_difference_product += abs(health1[r] - health2[c])
                    elif r < len(health1):
                        health_difference_product += abs(health1[r])  # Add the penalty for unmatched units.
                    elif c < len(health2):
                        health_difference_product += abs(health2[c])  # Add the penalty for unmatched units.

                return total_distance, health_difference_product

        # Compute blue-army coordinate and health differences.
        distance_blue, health_difference_blue = calculate_army_distance_and_health_difference(
            coords1_blue, coords2_blue, health1_blue, health2_blue, max_distance
        )
        total_distance += distance_blue
        total_health_difference += health_difference_blue

        # Compute red-army coordinate and health differences.
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
        """Return whether the coordinate distributions match and their health difference, using the configured threshold."""
        # Compute coordinate and health differences between states.
        distance_calculator = DistributionDistance(obs1, obs2)
        distribution_distance, health_distance = distance_calculator()  # Evaluate the DistributionDistance callable.
        return distribution_distance, health_distance