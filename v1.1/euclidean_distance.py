import numpy as np
import time
from multiprocessing import Pool
from collections import deque
import sys


def find_optimal_position(self, view_radius=10, avoid_edges=True, use_parallel=True):
    """
    Find the optimal standing position that maximizes visible spawn locations.
    Only considers positions that are actually walkable.

    :param view_radius: Maximum radius the player can see from their position (default: 10)
    :param avoid_edges: If True, avoid positions near the edge of the map
    :param use_parallel: If True, use parallel processing for faster computation
    :return: (x, y) coordinates of the optimal position and the number of visible spawns
    """
    # Create a coverage map to track how many spawn locations each position can see
    coverage_map = np.zeros_like(self.map, dtype=int)

    # Create a mask to avoid edges if requested
    edge_mask = np.ones_like(self.map, dtype=bool)
    if avoid_edges:
        # Create a margin that's at least the view radius
        margin = view_radius
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

    print("\nCalculating optimal position...")

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

    # Function to process a batch of positions
    def process_batch(positions_batch):
        batch_results = {}
        for x, y in positions_batch:
            if valid_mask[y, x]:
                # Count visible spawn locations from this position
                count = self._count_visible_spawns_optimized(x, y, view_radius)
                batch_results[(x, y)] = count
        return batch_results

    # Calculate batch size based on available CPU cores and memory
    import psutil
    from multiprocessing import cpu_count

    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    max_cores = min(cpu_count(), 8)  # Limit to 8 cores max

    # Adjust batch size based on available memory and map size
    estimated_memory_per_position = 0.01  # MB (estimated)
    max_positions_per_batch = int(
        available_memory / estimated_memory_per_position / max_cores / 2
    )
    batch_size = min(max(100, max_positions_per_batch), 5000)

    if use_parallel and total_positions > 1000:
        # Use parallel processing for large maps
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
                        percent_complete,
                        positions_checked,
                        total_positions,
                        start_time,
                    )
    else:
        # Sequential processing
        print("Using sequential processing")
        for i, (x, y) in enumerate(positions_to_check):
            # Count visible spawn locations from this position
            coverage_map[y, x] = self._count_visible_spawns_optimized(x, y, view_radius)

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

    # Find the position with maximum coverage (only among valid positions)
    if np.any(valid_mask):
        max_coverage = np.max(coverage_map[valid_mask])
        if max_coverage > 0:
            # Find all positions with max coverage
            max_positions = np.where((coverage_map == max_coverage) & valid_mask)
            if len(max_positions[0]) > 0:
                # Choose the position closest to the center if there are multiple options
                center_y, center_x = self.height // 2, self.width // 2
                min_dist = float("inf")
                best_idx = 0

                for i in range(len(max_positions[0])):
                    y, x = max_positions[0][i], max_positions[1][i]
                    dist = (y - center_y) ** 2 + (x - center_x) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                best_y, best_x = (
                    max_positions[0][best_idx],
                    max_positions[1][best_idx],
                )
                return best_x, best_y, max_coverage

    print("No valid positions found! Trying again without edge avoidance...")
    if avoid_edges:
        return self.find_optimal_position(
            view_radius, avoid_edges=False, use_parallel=use_parallel
        )
    return None, None, 0


def _count_visible_spawns(self, x, y, radius):
    """
    Original count method - kept for compatibility
    """
    spawn_count = 0

    # Check all locations within radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            check_x, check_y = x + dx, y + dy

            # Skip out-of-bounds
            if (
                check_x < 0
                or check_x >= self.width
                or check_y < 0
                or check_y >= self.height
            ):
                continue

            # Skip the central position itself
            if dx == 0 and dy == 0:
                continue

            # Check if within circular radius
            distance = np.sqrt(dx**2 + dy**2)
            if distance > radius:
                continue

            # Count if it's a spawn location
            if self.map[check_y, check_x] == 1:
                spawn_count += 1

    return spawn_count


def _count_visible_spawns_optimized(self, x, y, radius):
    """
    Optimized method to count visible spawn locations using OSRS-style pathfinding
    """
    # Calculate distances from this position to all spawn locations
    # Use the pre-calculated spawn_locations list for efficiency
    if len(self.spawn_locations) == 0:
        return 0

    # First, do a quick check using Euclidean distance for efficiency
    # This skips spawns that are definitely too far away
    distances = np.sqrt(((self.spawn_locations - [y, x]) ** 2).sum(axis=1))
    potential_visible = self.spawn_locations[distances <= radius]

    # For spawns within the radius, check if they're actually reachable
    visible_count = 0

    # Use BFS to find reachable tiles within radius
    reachable = self._find_reachable_tiles(x, y, radius)

    # Count spawn locations that are in reachable tiles
    for spawn_y, spawn_x in potential_visible:
        if (spawn_y, spawn_x) in reachable:
            visible_count += 1

    return visible_count


def _find_reachable_tiles(self, start_x, start_y, radius):
    """
    Find all tiles reachable from the starting position within the given radius
    using OSRS-style pathfinding (modified BFS with directional priorities)

    :param start_x: Starting X coordinate
    :param start_y: Starting Y coordinate
    :param radius: Maximum distance to check
    :return: Set of (y, x) coordinates of reachable tiles
    """
    # Use a queue for BFS
    queue = deque([(start_y, start_x)])
    # Track visited tiles
    visited = {(start_y, start_x): 0}  # (y, x): distance

    while queue:
        y, x = queue.popleft()
        current_distance = visited[(y, x)]

        # If we've reached the maximum distance, skip exploring further from this tile
        if current_distance >= radius:
            continue

        # Check neighbours in OSRS priority order (W, E, S, N, SW, SE, NW, NE)
        for dy, dx in self.directions:
            ny, nx = y + dy, x + dx

            # Skip if out of bounds
            if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                continue

            # Skip if not walkable or already visited
            if not self.walkable_map[ny, nx] or (ny, nx) in visited:
                continue

            # Calculate the new distance (diagonal moves count as sqrt(2) distance)
            new_distance = current_distance
            if dx != 0 and dy != 0:  # Diagonal move
                new_distance += np.sqrt(2)
            else:  # Cardinal move
                new_distance += 1

            # Skip if beyond radius
            if new_distance > radius:
                continue

            # Add to visited and queue
            visited[(ny, nx)] = new_distance
            queue.append((ny, nx))

    return visited
