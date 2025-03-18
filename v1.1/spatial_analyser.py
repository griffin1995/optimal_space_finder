import numpy as np
import time
import sys
import os
from collections import deque
import heapq
import psutil
from multiprocessing import Pool, cpu_count


def convert_to_game_coordinates(x, y):
    """
    Convert map coordinates to in-game coordinates.
    Replace with your actual conversion formula.
    :param x: Map x coordinate
    :param y: Map y coordinate
    :return: Tuple of (game_x, game_y)
    """
    # This is a placeholder - replace with your actual conversion formula
    game_x = x + 3000  # Example offset - REPLACE WITH ACTUAL VALUES
    game_y = y + 10000  # Example offset - REPLACE WITH ACTUAL VALUES
    return game_x, game_y


class SpawnOptimizer:
    def __init__(self, map_data, walkable_map=None):
        """
        Initialize the optimizer with a 2D map of possible spawn locations.
        :param map_data: 2D numpy array where 1 represents a valid spawn location, 0 represents invalid
        :param walkable_map: 2D numpy array where 1 represents walkable tiles, 0 represents unwalkable (walls, objects)
        If None, assumes all non-spawn locations are walkable
        """
        self.map = np.array(map_data)
        self.height, self.width = self.map.shape

        # Set up walkable map (for pathfinding)
        if walkable_map is None:
            # If no walkable map provided, assume all tiles are walkable
            self.walkable_map = np.ones_like(self.map)
        else:
            self.walkable_map = np.array(walkable_map)

        # Pre-calculate spawn locations for faster access
        self.spawn_locations = np.array(np.where(self.map == 1)).T
        self.total_spawns = len(self.spawn_locations)

        # Directions for pathfinding (in order of priority)
        # OSRS checks in this order: W, E, S, N, SW, SE, NW, NE
        self.directions = [
            (0, -1),  # West
            (0, 1),  # East
            (1, 0),  # South
            (-1, 0),  # North
            (1, -1),  # Southwest
            (1, 1),  # Southeast
            (-1, -1),  # Northwest
            (-1, 1),  # Northeast
        ]

        print(f"Map dimensions: {self.width}x{self.height}")
        print(f"Total spawn locations: {self.total_spawns}")
        print(f"Walkable tiles: {np.sum(self.walkable_map)}")

        # Calculate memory usage
        memory_usage = self.map.nbytes / (1024 * 1024)  # MB
        print(f"Memory usage for map: {memory_usage:.2f} MB")

    def update_progress(
        self, percent_complete, positions_checked, total_positions, start_time
    ):
        """
        Update the progress display
        """
        bars = int(percent_complete / 10)
        progress_bar = "[" + "#" * bars + " " * (10 - bars) + "]"
        elapsed = time.time() - start_time
        positions_per_second = positions_checked / max(1, elapsed)
        remaining_time = (total_positions - positions_checked) / max(
            1, positions_per_second
        )

        # Format time as h:m:s
        remaining_min, remaining_sec = divmod(remaining_time, 60)
        remaining_hour, remaining_min = divmod(remaining_min, 60)
        time_str = f"{int(remaining_hour):02d}:{int(remaining_min):02d}:{int(remaining_sec):02d}"

        print(
            f"{progress_bar} {percent_complete}% complete - ETA: {time_str} - {positions_per_second:.1f} pos/sec",
            end="\r",
        )
        sys.stdout.flush()

    def find_optimal_position_tile_distance(self, max_tiles=10, use_parallel=False):
        """
        Find the optimal position using tile distance method

        :param max_tiles: Maximum number of tiles to consider
        :param use_parallel: Whether to use parallel processing
        :return: Tuple of (optimal_x, optimal_y, total_coverage)
        """
        from tile_distance import find_optimal_position_tile_distance

        return find_optimal_position_tile_distance(
            self.map, self.walkable_map, max_tiles=max_tiles, use_parallel=use_parallel
        )

    def find_optimal_position(self, view_radius=10, use_parallel=False):
        """
        Find the optimal position using Euclidean distance method

        :param view_radius: Radius of view
        :param use_parallel: Whether to use parallel processing
        :return: Tuple of (optimal_x, optimal_y, total_coverage)
        """
        from euclidean_distance import find_optimal_position

        return find_optimal_position(
            self.map,
            self.walkable_map,
            view_radius=view_radius,
            use_parallel=use_parallel,
        )
