# spawn_testing.py

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import Counter
from spatial_analyser import convert_to_game_coordinates


class SpawnTester:
    def __init__(self, optimizer):
        """
        Initialize the spawn tester with a reference to the optimizer

        :param optimizer: SpawnOptimizer instance
        """
        self.optimizer = optimizer
        self.test_position = None  # The position where testing was done
        self.observed_spawns = []  # List of (x, y) coordinates of observed spawns
        self.spawn_counts = {}  # Dictionary of (x, y): count for multiple spawns
        self.max_observed_distance = 0  # Maximum distance of successful interaction

    def load_test_data(self, filename):
        """
        Load spawn test data from a CSV file

        Expected format: x,y,count (where count is number of spawns observed at that location)

        :param filename: Path to the CSV file
        :return: True if successful, False otherwise
        """
        try:
            with open(filename, "r") as f:
                reader = csv.reader(f)
                # Skip header if present
                if next(reader)[0].lower() in ["x", "tile_x", "spawn_x"]:
                    pass  # Skip header
                else:
                    # Reset file pointer and read again if no header
                    f.seek(0)
                    reader = csv.reader(f)

                # Read spawn data
                self.observed_spawns = []
                self.spawn_counts = {}

                for row in reader:
                    if len(row) >= 2:
                        x, y = int(row[0]), int(row[1])
                        count = int(row[2]) if len(row) >= 3 else 1

                        self.observed_spawns.extend([(x, y)] * count)
                        self.spawn_counts[(x, y)] = count

            print(
                f"Loaded {len(self.observed_spawns)} spawn observations at {len(self.spawn_counts)} unique locations"
            )
            return True
        except Exception as e:
            print(f"Error loading spawn test data: {e}")
            return False

    def set_test_position(self, x, y):
        """
        Set the position where testing was conducted

        :param x: X coordinate of test position
        :param y: Y coordinate of test position
        """
        self.test_position = (x, y)
        print(f"Set test position to ({x}, {y})")

    def calculate_distances(self):
        """
        Calculate distances from test position to all observed spawns

        :return: Dictionary of distances and list of (spawn_loc, distance) pairs
        """
        if not self.test_position:
            print("Error: Test position not set")
            return None, []

        distances = []
        for x, y in self.observed_spawns:
            # Calculate tile distance (using pathfinding)
            if hasattr(self.optimizer, "_find_reachable_tiles_tile_distance"):
                reachable = self.optimizer._find_reachable_tiles_tile_distance(
                    self.test_position[0], self.test_position[1], 20
                )  # Assuming 20 is a safe maximum
                if (y, x) in reachable:
                    dist = reachable[(y, x)]
                else:
                    # If not reachable, use Manhattan distance as fallback
                    dist = abs(x - self.test_position[0]) + abs(
                        y - self.test_position[1]
                    )
            else:
                # Fallback to Manhattan distance
                dist = abs(x - self.test_position[0]) + abs(y - self.test_position[1])

            distances.append(dist)

        # Calculate the maximum observed distance (successful interaction)
        if distances:
            self.max_observed_distance = max(distances)

        # Count observations at each distance
        distance_counts = Counter(distances)

        # Create a list of (spawn_location, distance) pairs
        spawn_distances = [
            (loc, self._calculate_single_distance(loc, self.test_position))
            for loc in self.spawn_counts.keys()
        ]

        return distance_counts, spawn_distances

    def _calculate_single_distance(self, spawn_loc, test_pos):
        """Calculate distance between a spawn location and test position"""
        x, y = spawn_loc
        test_x, test_y = test_pos

        # Use pathfinding if available
        if hasattr(self.optimizer, "_find_reachable_tiles_tile_distance"):
            reachable = self.optimizer._find_reachable_tiles_tile_distance(
                test_x, test_y, 20
            )
            if (y, x) in reachable:
                return reachable[(y, x)]

        # Fallback to Manhattan distance
        return abs(x - test_x) + abs(y - test_y)

    def visualize_observations(self):
        """
        Create a visualization of observed spawn locations and distances
        """
        if not self.test_position or not self.observed_spawns:
            print("Error: Test position not set or no spawn observations loaded")
            return None

        # Calculate distances
        distance_counts, spawn_distances = self.calculate_distances()

        # Create a figure
        plt.figure(figsize=(15, 12))

        # Create a background map
        visualization = np.zeros(
            (self.optimizer.height, self.optimizer.width, 3), dtype=np.uint8
        )

        # Set the walkable area to dark gray
        walkable_mask = self.optimizer.walkable_map > 0
        visualization[walkable_mask] = [50, 50, 50]

        # Create a heatmap of spawn frequencies
        # heatmap = np.zeros_like(self.optimizer.map, dtype=int)
        # for (x, y), count in self.spawn_counts.items():
        #     if 0 <= y < self.optimizer.height and 0 <= x < self.optimizer.width:
        #         heatmap[y, x] = count

        # Add the background to the plot
        plt.imshow(visualization)

        # Mark the test position
        test_x, test_y = self.test_position
        plt.plot(test_x, test_y, "r*", markersize=15)

        # Convert game coordinates
        game_x, game_y = convert_to_game_coordinates(test_x, test_y)

        # Draw circles for different distances from test position
        # Draw circles from max observed distance down to 1
        for radius in range(int(self.max_observed_distance), 0, -1):
            circle = plt.Circle(
                (test_x, test_y),
                radius,
                color="cyan",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            plt.gca().add_patch(circle)

        # Mark observed spawn locations with size based on count
        for (x, y), count in self.spawn_counts.items():
            # Calculate distance
            dist = self._calculate_single_distance((x, y), self.test_position)

            # Color based on distance (within max_observed_distance or not)
            color = "green" if dist <= self.max_observed_distance else "red"

            # Size based on count, but with a minimum size
            size = max(30, count * 10)

            # Plot the spawn location
            plt.scatter(x, y, s=size, color=color, alpha=0.7, edgecolors="white")

            # Add label with count if more than 1
            if count > 1:
                plt.text(
                    x + 1,
                    y + 1,
                    str(count),
                    color="white",
                    fontsize=8,
                    backgroundcolor="black",
                    ha="center",
                    va="center",
                )

        # Add a legend
        plt.scatter(
            [],
            [],
            s=30,
            color="green",
            alpha=0.7,
            edgecolors="white",
            label="Within observed radius",
        )
        plt.scatter(
            [],
            [],
            s=30,
            color="red",
            alpha=0.7,
            edgecolors="white",
            label="Beyond observed radius",
        )
        plt.scatter(
            [],
            [],
            s=30,
            color="blue",
            alpha=0.7,
            edgecolors="white",
            label="Optimized position",
        )

        # Add title and legend
        plt.title(
            f"Observed Spawn Locations from Test Position ({test_x}, {test_y})\n"
            f"Game coordinates: ({game_x}, {game_y})\n"
            f"Max observed interaction distance: {self.max_observed_distance} tiles"
        )
        plt.legend()

        # Save the visualization
        output_path = "observed_spawns_visualization.png"
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")

        return plt

    def compare_with_optimal(self, optimal_position, optimal_radius):
        """
        Compare the empirical testing results with the computational optimal position

        :param optimal_position: (x, y) tuple of the optimal position from computation
        :param optimal_radius: The radius used in optimization
        """
        if not self.test_position or not self.observed_spawns:
            print("Error: Test position not set or no spawn observations loaded")
            return None

        # Create a figure
        plt.figure(figsize=(15, 12))

        # Create a background map
        visualization = np.zeros(
            (self.optimizer.height, self.optimizer.width, 3), dtype=np.uint8
        )

        # Set the walkable area to dark gray
        walkable_mask = self.optimizer.walkable_map > 0
        visualization[walkable_mask] = [50, 50, 50]

        # Add the background to the plot
        plt.imshow(visualization)

        # Mark the test position
        test_x, test_y = self.test_position
        plt.plot(test_x, test_y, "r*", markersize=15)
        game_test_x, game_test_y = convert_to_game_coordinates(test_x, test_y)

        # Mark the optimal position
        opt_x, opt_y = optimal_position
        plt.plot(opt_x, opt_y, "b*", markersize=15)
        game_opt_x, game_opt_y = convert_to_game_coordinates(opt_x, opt_y)

        # Draw empirical radius around test position
        empirical_circle = plt.Circle(
            (test_x, test_y),
            self.max_observed_distance,
            color="red",
            fill=False,
            linestyle="-.",
        )
        plt.gca().add_patch(empirical_circle)

        # Draw computational radius around optimal position
        optimal_circle = plt.Circle(
            (opt_x, opt_y), optimal_radius, color="blue", fill=False, linestyle="--"
        )
        plt.gca().add_patch(optimal_circle)

        # Mark observed spawn locations
        for (x, y), count in self.spawn_counts.items():
            # Size based on count
            size = max(20, count * 8)
            plt.scatter(x, y, s=size, color="green", alpha=0.7, edgecolors="white")

        # Add legend and title
        plt.title(
            f"Comparison of Test Position vs. Optimal Position\n"
            f"Test: ({test_x}, {test_y}) Game: ({game_test_x}, {game_test_y}) - "
            f"Empirical Radius: {self.max_observed_distance}\n"
            f"Optimal: ({opt_x}, {opt_y}) Game: ({game_opt_x}, {game_opt_y}) - "
            f"Computational Radius: {optimal_radius}"
        )

        # Add a legend
        plt.plot([], [], "r*", markersize=10, label="Test Position")
        plt.plot([], [], "b*", markersize=10, label="Optimal Position")
        plt.plot(
            [],
            [],
            "r-.",
            label=f"Empirical Radius ({self.max_observed_distance} tiles)",
        )
        plt.plot([], [], "b--", label=f"Computational Radius ({optimal_radius} tiles)")
        plt.legend()

        # Save the visualization
        output_path = "test_vs_optimal_comparison.png"
        plt.savefig(output_path)
        print(f"Saved comparison to {output_path}")

        return plt

    def generate_report(self, output_file="spawn_testing_report.txt"):
        """
        Generate a report of the empirical testing results

        :param output_file: Path to the output file
        """
        if not self.test_position or not self.observed_spawns:
            print("Error: Test position not set or no spawn observations loaded")
            return False

        # Calculate distances
        distance_counts, spawn_distances = self.calculate_distances()

        with open(output_file, "w") as f:
            f.write("SEAWEED SPORE SPAWN TESTING REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Test position info
            test_x, test_y = self.test_position
            game_x, game_y = convert_to_game_coordinates(test_x, test_y)
            f.write(
                f"Test Position: ({test_x}, {test_y}) - Game coordinates: ({game_x}, {game_y})\n"
            )
            f.write(f"Total observed spawns: {len(self.observed_spawns)}\n")
            f.write(f"Unique spawn locations: {len(self.spawn_counts)}\n")
            f.write(
                f"Maximum observed interaction distance: {self.max_observed_distance} tiles\n\n"
            )

            # Distribution of spawns by distance
            f.write("SPAWN DISTRIBUTION BY DISTANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Distance':<10}{'Count':<10}{'Percentage':<15}\n")

            # Calculate percentages
            total_spawns = len(self.observed_spawns)

            # Sort by distance
            for distance in sorted(distance_counts.keys()):
                count = distance_counts[distance]
                percentage = (count / total_spawns) * 100
                f.write(f"{distance:<10}{count:<10}{percentage:.2f}%\n")

            f.write("\n")

            # List of all spawn locations with counts and distances
            f.write("DETAILED SPAWN LOCATIONS\n")
            f.write("-" * 60 + "\n")
            f.write(
                f"{'X':<5}{'Y':<5}{'Game X':<10}{'Game Y':<10}{'Count':<8}{'Distance':<10}\n"
            )

            # Sort by distance
            sorted_spawns = sorted(spawn_distances, key=lambda x: x[1])

            for (x, y), distance in sorted_spawns:
                count = self.spawn_counts[(x, y)]
                game_x, game_y = convert_to_game_coordinates(x, y)
                f.write(
                    f"{x:<5}{y:<5}{game_x:<10}{game_y:<10}{count:<8}{distance:<10}\n"
                )

        print(f"Generated report at {output_file}")
        return True


def add_spawn_testing_to_main(
    input_file=None,
    test_position=None,
    optimal_position=None,
    optimal_radius=None,
    optimizer=None,  # Add this parameter
):
    """
    Function to be called from main.py to add spawn testing functionality
    """
    # If optimizer is not provided, try to import from main
    if optimizer is None:
        try:
            from main import optimizer
        except ImportError:
            print("Error: Could not import optimizer from main.py")
            return None
