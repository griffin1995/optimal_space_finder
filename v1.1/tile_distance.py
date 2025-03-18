import numpy as np
import time
import heapq
from multiprocessing import Pool, cpu_count
import psutil


def _find_reachable_tiles_tile_distance(self, start_x, start_y, max_tile_distance):
    """
    Find all tiles reachable from the starting position within the given max tile distance
    using game-accurate pathfinding (shortest path in number of tiles)

    :param start_x: Starting X coordinate
    :param start_y: Starting Y coordinate
    :param max_tile_distance: Maximum number of tiles to travel
    :return: Dictionary of (y, x): distance for reachable tiles
    """
    # Use a priority queue for Dijkstra's algorithm
    # The queue will store (distance, y, x) tuples
    priority_queue = [(0, start_y, start_x)]
    # Track visited tiles and their distances
    distances = {(start_y, start_x): 0}  # (y, x): distance

    # Store final reachable tiles
    reachable = {}

    while priority_queue:
        # Get the position with lowest distance (Dijkstra's algorithm)
        current_distance, y, x = heapq.heappop(priority_queue)

        # Skip if we've found a better path already
        if current_distance > distances.get((y, x), float("inf")):
            continue

        # If we've reached the maximum tile distance, don't explore further
        if current_distance > max_tile_distance:
            continue

        # Add to final reachable set
        reachable[(y, x)] = current_distance

        # Check all 8 directions
        for dy, dx in self.directions:
            ny, nx = y + dy, x + dx

            # Skip if out of bounds
            if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                continue

            # Skip if not walkable
            if not self.walkable_map[ny, nx]:
                continue

            # Calculate new distance (tile distance)
            # Every step costs 1 tile, regardless of whether it's diagonal or cardinal
            new_distance = current_distance + 1

            # Skip if beyond max distance
            if new_distance > max_tile_distance:
                continue

            # Skip if we already found a better path
            if new_distance >= distances.get((ny, nx), float("inf")):
                continue

            # Update distance and add to queue
            distances[(ny, nx)] = new_distance
            heapq.heappush(priority_queue, (new_distance, ny, nx))

    return reachable


def _count_visible_spawns_tile_distance(self, x, y, max_tiles):
    """
    Count spawn locations within a given number of tiles (using shortest path)

    :param x: X coordinate
    :param y: Y coordinate
    :param max_tiles: Maximum number of tiles to travel
    :return: Number of spawn locations visible
    """
    # Find all reachable tiles within max_tiles using tile distance
    reachable = self._find_reachable_tiles_tile_distance(x, y, max_tiles)

    # Count spawn locations that are in reachable tiles
    visible_count = 0
    for spawn_y, spawn_x in self.spawn_locations:
        if (spawn_y, spawn_x) in reachable:
            visible_count += 1

    return visible_count


def find_optimal_position_tile_distance(
    self, max_tiles=10, avoid_edges=True, use_parallel=True
):
    """
    Find the optimal standing position that maximizes visible spawn locations,
    using tile distance rather than Euclidean distance.

    :param max_tiles: Maximum number of tiles the player can travel from their position
    :param avoid_edges: If True, avoid positions near the edge of the map
    :param use_parallel: If True, use parallel processing for faster computation
    :return: (x, y) coordinates of the optimal position and the number of visible spawns
    """
    # Create a coverage map to track how many spawn locations each position can see
    coverage_map = np.zeros_like(self.map, dtype=int)

    # Create a mask to avoid edges if requested
    edge_mask = np.ones_like(self.map, dtype=bool)
    if avoid_edges:
        # Create a margin that's at least the max_tiles
        margin = max_tiles
        edge_mask[:margin, :] = False  # Top margin
        edge_mask[-margin:, :] = False  # Bottom margin
        edge_mask[:, :margin] = False  # Left margin
        edge_mask[:, -margin:] = False  # Right margin
        print(f"Avoiding positions within {margin} tiles of the map edge")

    # Only consider walkable positions
    walkable_mask = self.walkable_map > 0
    valid_mask = edge_mask & walkable_mask
    print(f"Total valid positions to check: {np.sum(valid_mask)}")

    # Calculate total positions to check for progress reporting
    valid_positions = np.sum(valid_mask)
    start_time = time.time()

    print("\nCalculating optimal position using tile distance...")

    # Check if the map is too large for full computation
    if self.width * self.height > 1000000:  # For very large maps
        print("Large map detected. Using sampling approach...")
        # Sample every N positions
        sample_factor = max(2, min(10, int(np.sqrt(self.width * self.height) / 300)))
        print(f"Sampling every {sample_factor} positions")

        positions_to_check = []
        for y in range(0, self.height, sample_factor):
            for x in range(0, self.width, sample_factor):
                if valid_mask[y, x]:
                    positions_to_check.append((x, y))
    else:
        # Check all valid positions
        positions_to_check = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if valid_mask[y, x]
        ]

    # Progress tracking variables
    positions_checked = 0
    last_percent = 0
    total_positions = len(positions_to_check)
    print(f"Checking {total_positions} positions...")

    # Function to process a batch of positions using tile distance
    def process_batch(positions_batch):
        batch_results = {}
        for x, y in positions_batch:
            # Count visible spawn locations using tile distance
            count = self._count_visible_spawns_tile_distance(x, y, max_tiles)
            batch_results[(x, y)] = count
        return batch_results

    # If using parallel processing
    if use_parallel and total_positions > 1000:
        # Calculate batch size based on available CPU cores and memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        max_cores = min(cpu_count(), 8)  # Limit to 8 cores max

        # Adjust batch size based on available memory and map size
        estimated_memory_per_position = 0.01  # MB (estimated)
        max_positions_per_batch = int(
            available_memory / estimated_memory_per_position / max_cores / 2
        )
        batch_size = min(max(100, max_positions_per_batch), 5000)

        print(
            f"Using parallel processing with {max_cores} cores, batch size: {batch_size}"
        )

        # Split positions into batches
        position_batches = [
            positions_to_check[i : i + batch_size]
            for i in range(0, len(positions_to_check), batch_size)
        ]

        # Process batches in parallel
        with Pool(processes=max_cores) as pool:
            # Map batches to worker processes
            for i, batch_results in enumerate(
                pool.imap_unordered(process_batch, position_batches)
            ):
                # Update coverage map with batch results
                for (x, y), count in batch_results.items():
                    coverage_map[y, x] = count

                # Update progress
                positions_checked += len(batch_results)
                percent_complete = int((i + 1) / len(position_batches) * 100)

                # Only update display when percentage changes
                if percent_complete > last_percent:
                    last_percent = percent_complete
                    self._update_progress(
                        percent_complete, positions_checked, total_positions, start_time
                    )
    else:
        # Sequential processing
        print("Using sequential processing")
        for i, (x, y) in enumerate(positions_to_check):
            # Count visible spawn locations from this position
            coverage_map[y, x] = self._count_visible_spawns_tile_distance(
                x, y, max_tiles
            )

            # Update progress
            positions_checked += 1
            percent_complete = int((positions_checked / total_positions) * 100)

            # Only update display when percentage changes
            if percent_complete > last_percent:
                last_percent = percent_complete
                self._update_progress(
                    percent_complete, positions_checked, total_positions, start_time
                )

    # Clear the progress line and print completion
    print(" " * 80, end="\r")  # Clear the line
    elapsed = time.time() - start_time
    print(f"Calculation complete in {elapsed:.2f} seconds!")

    # Find the position with maximum coverage (excluding edges)
    # Use a mask to only consider valid positions
    max_coverage = 0
    best_x, best_y = None, None

    for y in range(self.height):
        for x in range(self.width):
            if valid_mask[y, x] and coverage_map[y, x] > max_coverage:
                max_coverage = coverage_map[y, x]
                best_x, best_y = x, y

    if best_x is not None and best_y is not None:
        print(
            f"Found optimal position at ({best_x}, {best_y}) with {max_coverage} visible spawns"
        )
        return best_x, best_y, max_coverage
    else:
        print("No valid positions found! Trying again without edge avoidance...")
        if avoid_edges:
            return self.find_optimal_position_tile_distance(
                max_tiles, avoid_edges=False, use_parallel=use_parallel
            )
        return None, None, 0
