import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import argparse
import time
import sys
import os
from multiprocessing import Pool, cpu_count
import psutil
from collections import deque
import heapq


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

        # Directions for OSRS pathfinding (in order of priority)
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

    def find_optimal_position(
        self, view_radius=10, avoid_edges=True, use_parallel=True
    ):
        """
        Find the optimal standing position that maximizes visible spawn locations.

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

        # Calculate total positions to check for progress reporting
        valid_positions = np.sum(edge_mask)
        start_time = time.time()

        print("\nCalculating optimal position...")

        # Check if the map is too large for full computation
        if self.width * self.height > 1000000:  # For very large maps
            print("Large map detected. Using sampling approach...")
            # Sample every N positions
            sample_factor = max(
                2, min(10, int(np.sqrt(self.width * self.height) / 300))
            )
            print(f"Sampling every {sample_factor} positions")

            positions_to_check = []
            for y in range(0, self.height, sample_factor):
                for x in range(0, self.width, sample_factor):
                    if edge_mask[y, x]:
                        positions_to_check.append((x, y))
        else:
            # Check all positions
            positions_to_check = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if edge_mask[y, x]
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
                if edge_mask[y, x]:
                    # Count visible spawn locations from this position
                    count = self._count_visible_spawns_optimized(x, y, view_radius)
                    batch_results[(x, y)] = count
            return batch_results

        # Calculate batch size based on available CPU cores and memory
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
                coverage_map[y, x] = self._count_visible_spawns_optimized(
                    x, y, view_radius
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
        # Use a mask to ignore -1 values (edge positions)
        valid_mask = coverage_map >= 0
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

    def _update_progress(
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

        # For spawns within the radius, check if they're actually reachable using OSRS pathfinding
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

    def calculate_best_positions_by_sector(self, view_radius=10, sectors=4):
        """
        Find best positions by dividing the map into sectors

        :param view_radius: View radius
        :param sectors: Number of sectors (square root of total sectors)
        :return: List of best positions per sector with coverage
        """
        print(f"Analyzing map by dividing into {sectors}x{sectors} sectors...")

        # Divide map into sectors
        sector_height = self.height // sectors
        sector_width = self.width // sectors

        best_positions = []

        for sy in range(sectors):
            for sx in range(sectors):
                # Calculate sector boundaries
                y_start = sy * sector_height
                y_end = min((sy + 1) * sector_height, self.height)
                x_start = sx * sector_width
                x_end = min((sx + 1) * sector_width, self.width)

                print(
                    f"Analyzing sector ({sx},{sy}): ({x_start},{y_start}) to ({x_end},{y_end})"
                )

                # Create a sub-map for this sector
                sector_map = np.zeros_like(self.map)
                sector_map[y_start:y_end, x_start:x_end] = self.map[
                    y_start:y_end, x_start:x_end
                ]

                # Find optimal position within this sector
                best_coverage = 0
                best_pos = None

                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        # Only check valid positions (not too close to sector edge)
                        if (
                            x >= x_start + view_radius // 2
                            and x < x_end - view_radius // 2
                            and y >= y_start + view_radius // 2
                            and y < y_end - view_radius // 2
                        ):
                            coverage = self._count_visible_spawns_optimized(
                                x, y, view_radius
                            )
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_pos = (x, y)

                if best_pos:
                    best_positions.append((best_pos[0], best_pos[1], best_coverage))
                    print(
                        f"Best position in sector ({sx},{sy}): {best_pos} with {best_coverage} visible spawns"
                    )

        # Sort by coverage
        best_positions.sort(key=lambda x: x[2], reverse=True)
        return best_positions

    def visualize_coverage(
        self, optimal_position=None, view_radius=10, sample_step=None
    ):
        """
        Create a heatmap visualization of the spawn coverage.

        :param optimal_position: Optional tuple of (x, y) for marking optimal position
        :param view_radius: View radius
        :param sample_step: Step size for sampling (None for automatic)
        """
        print("\nGenerating coverage heatmap...")
        start_time = time.time()

        plt.figure(figsize=(15, 12))

        # Create coverage map
        coverage_map = np.zeros_like(self.map, dtype=int)

        # Use smaller step size for large maps to speed up visualization
        if sample_step is None:
            sample_step = max(1, min(5, self.width // 200))
        print(f"Using step size of {sample_step} for visualization sampling")

        # Calculate total positions for progress reporting
        total_positions = (self.height // sample_step) * (self.width // sample_step)
        positions_checked = 0
        last_percent = 0

        print("[          ] 0% complete", end="\r")

        for y in range(0, self.height, sample_step):
            for x in range(0, self.width, sample_step):
                coverage_map[y, x] = self._count_visible_spawns_optimized(
                    x, y, view_radius
                )

                # Update progress
                positions_checked += 1
                percent_complete = int((positions_checked / total_positions) * 100)

                # Only update display when percentage changes
                if percent_complete > last_percent:
                    last_percent = percent_complete
                    bars = int(percent_complete / 10)
                    progress_bar = "[" + "#" * bars + " " * (10 - bars) + "]"
                    print(f"{progress_bar} {percent_complete}% complete", end="\r")
                    sys.stdout.flush()

        # Clear the progress line
        print(" " * 80, end="\r")

        # Upsample coverage map to full resolution if sampled
        if sample_step > 1:
            # Use nearest-neighbor upsampling for coverage map
            full_coverage = np.zeros_like(self.map, dtype=int)
            for y in range(self.height):
                for x in range(self.width):
                    sample_y = min(
                        (y // sample_step) * sample_step, coverage_map.shape[0] - 1
                    )
                    sample_x = min(
                        (x // sample_step) * sample_step, coverage_map.shape[1] - 1
                    )
                    full_coverage[y, x] = coverage_map[sample_y, sample_x]
            coverage_map = full_coverage

        # Create heatmap of the coverage
        print("Drawing heatmap...")

        # Use a better colormap
        ax = sns.heatmap(
            coverage_map,
            cmap="viridis",  # Blue-Green-Yellow colormap
            cbar=True,
            square=True,
            annot=False,
            fmt="d",
            linewidths=0,
        )

        # Add spawn location markers
        spawn_y, spawn_x = np.where(self.map == 1)
        plt.scatter(
            spawn_x + 0.5, spawn_y + 0.5, color="red", s=10, alpha=0.5, marker="o"
        )

        # Mark optimal position if provided
        if optimal_position:
            x, y = optimal_position
            plt.plot(x + 0.5, y + 0.5, "r*", markersize=15)
            plt.title(
                f"Spawn Coverage Map (Optimal Position: {x},{y} with {coverage_map[y,x]} visible spawns)"
            )
        else:
            plt.title("Spawn Coverage Map")

        plt.tight_layout()

        # Save the visualization
        heatmap_path = "optimal_position_heatmap.png"
        plt.savefig(heatmap_path)
        print(f"Saved heatmap to {heatmap_path}")

        elapsed = time.time() - start_time
        print(f"Heatmap generated in {elapsed:.2f} seconds!")

        print("Displaying visualization...")
        plt.show()

        return coverage_map

    def visualize_map_with_optimal(
        self,
        optimal_position=None,
        view_radius=10,
        mark_alternatives=False,
        alternative_positions=None,
    ):
        """
        Visualize the map with spawn locations and optimal position marked.

        :param optimal_position: Optional tuple of (x, y) for marking optimal position
        :param view_radius: View radius
        :param mark_alternatives: If True, mark alternative good positions
        :param alternative_positions: List of alternative positions to mark
        """
        print("\nGenerating map visualization...")
        start_time = time.time()

        plt.figure(figsize=(15, 12))

        # Create a colored visualization
        visualization = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Set spawn locations to red
        mask = self.map == 1
        visualization[mask] = [255, 0, 0]  # Red for spawn locations

        plt.imshow(visualization)

        # Mark optimal position if provided
        if optimal_position:
            x, y = optimal_position
            plt.plot(x, y, "b*", markersize=15)
            plt.title(f"Spawn Map with Optimal Position ({x},{y})")

            # Draw the view radius circle
            circle = plt.Circle(
                (x, y), view_radius, color="b", fill=False, linestyle="--"
            )
            plt.gca().add_patch(circle)

            print("Calculating visible spawns from optimal position...")
            # Draw the coverage area - visible spawns
            visible_count = 0
            for dy in range(-view_radius, view_radius + 1):
                for dx in range(-view_radius, view_radius + 1):
                    check_x, check_y = x + dx, y + dy

                    # Skip out-of-bounds
                    if (
                        check_x < 0
                        or check_x >= self.width
                        or check_y < 0
                        or check_y >= self.height
                    ):
                        continue

                    # Check if within circular radius
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > view_radius:
                        continue

                    # Highlight visible spawn locations
                    if self.map[check_y, check_x] == 1:
                        plt.plot(
                            check_x, check_y, "go", markersize=5
                        )  # Small green circles
                        visible_count += 1

            print(f"Visible spawn locations from optimal position: {visible_count}")

            # Mark alternative positions if requested
            if mark_alternatives and alternative_positions:
                for i, (alt_x, alt_y, coverage) in enumerate(
                    alternative_positions[:5]
                ):  # Show top 5
                    plt.plot(alt_x, alt_y, "ys", markersize=10)
                    plt.text(
                        alt_x + 2,
                        alt_y + 2,
                        f"#{i+1}: {coverage}",
                        color="yellow",
                        fontsize=8,
                    )

        else:
            plt.title("Spawn Map")

        # Save the visualization
        map_path = "optimal_position_map.png"
        plt.savefig(map_path)
        print(f"Saved map visualization to {map_path}")

        elapsed = time.time() - start_time
        print(f"Map visualization generated in {elapsed:.2f} seconds!")

        plt.tight_layout()
        print("Displaying visualization...")
        plt.show()

    def create_highlighted_tile_map(
        self,
        optimal_position,
        view_radius=10,
        original_image_path=None,
        alternative_positions=None,
        pathfinding=False,
    ):
        """
        Create a map with clear highlighting of each potential spawn tile and the optimal position.

        :param optimal_position: Tuple of (x, y) for the optimal position
        :param view_radius: View radius
        :param original_image_path: Path to the original image to overlay on
        :param alternative_positions: List of alternative positions to mark
        :param pathfinding: If True, show paths between optimal positions
        """
        if optimal_position is None or optimal_position[0] is None:
            print("No optimal position to mark!")
            return

        print("\nCreating highlighted tile map...")

        # If original image is provided, use it as background
        if original_image_path and os.path.exists(original_image_path):
            try:
                background = Image.open(original_image_path)
                background = background.convert("RGB")
                img_array = np.array(background)
                print(f"Using original image as background with size {background.size}")
            except Exception as e:
                print(f"Error loading original image: {e}")
                img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Set dark gray background
                img_array[:, :] = [30, 30, 30]
        else:
            # Create a new image with dark background
            img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img_array[:, :] = [30, 30, 30]  # Dark gray background

        # Create a PIL Image for drawing
        image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(image)

        # Get the optimal position
        opt_x, opt_y = optimal_position

        # Draw grid if the map is not too large
        if self.width * self.height <= 1000000:  # Limit grid drawing for large maps
            print("Drawing grid...")
            # Draw grid lines (light gray)
            for x in range(0, self.width, 10):
                draw.line([(x, 0), (x, self.height - 1)], fill=(70, 70, 70), width=1)
            for y in range(0, self.height, 10):
                draw.line([(0, y), (self.width - 1, y)], fill=(70, 70, 70), width=1)

        # Highlight all potential spawn locations
        print("Highlighting all potential spawn locations...")
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y, x] == 1:
                    # Draw a red square for each spawn location
                    draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))

        # Calculate visible spawns from the optimal position
        print("Marking visible spawns from optimal position...")
        visible_spawns = []
        for dy in range(-view_radius, view_radius + 1):
            for dx in range(-view_radius, view_radius + 1):
                check_x, check_y = opt_x + dx, opt_y + dy

                # Skip out-of-bounds
                if (
                    check_x < 0
                    or check_x >= self.width
                    or check_y < 0
                    or check_y >= self.height
                ):
                    continue

                # Check if within circular radius
                distance = np.sqrt(dx**2 + dy**2)
                if distance > view_radius:
                    continue

                # Collect visible spawn locations
                if self.map[check_y, check_x] == 1:
                    visible_spawns.append((check_x, check_y))

        # Draw visible spawn locations in green
        for x, y in visible_spawns:
            draw.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 255, 0))

        # Draw the view radius circle
        print("Drawing view radius...")
        for angle in range(0, 360, 1):
            radian = angle * np.pi / 180
            x = opt_x + view_radius * np.cos(radian)
            y = opt_y + view_radius * np.sin(radian)

            # Check bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                # Draw cyan pixel on the circle
                draw.point((x, y), fill=(0, 255, 255))

        # Draw a very clear marker at the optimal position
        print("Marking optimal position...")
        marker_size = 10
        # Yellow circle
        draw.ellipse(
            [
                (opt_x - marker_size, opt_y - marker_size),
                (opt_x + marker_size, opt_y + marker_size),
            ],
            outline=(255, 255, 0),
            fill=(255, 255, 0),
            width=2,
        )

        # Blue X
        draw.line(
            [
                (opt_x - marker_size, opt_y - marker_size),
                (opt_x + marker_size, opt_y + marker_size),
            ],
            fill=(0, 0, 255),
            width=3,
        )
        draw.line(
            [
                (opt_x - marker_size, opt_y + marker_size),
                (opt_x + marker_size, opt_y - marker_size),
            ],
            fill=(0, 0, 255),
            width=3,
        )

        # Mark alternative positions if provided
        if alternative_positions:
            print("Marking alternative positions...")
            for i, (alt_x, alt_y, coverage) in enumerate(
                alternative_positions[:5]
            ):  # Top 5
                # Draw a smaller purple circle
                alt_marker_size = 6
                draw.ellipse(
                    [
                        (alt_x - alt_marker_size, alt_y - alt_marker_size),
                        (alt_x + alt_marker_size, alt_y + alt_marker_size),
                    ],
                    outline=(255, 0, 255),
                    fill=(255, 0, 255),
                    width=1,
                )

                # Add label
                try:
                    font = ImageFont.load_default()
                    draw.text(
                        (alt_x + alt_marker_size + 2, alt_y - alt_marker_size),
                        f"#{i+1}: {coverage}",
                        fill=(255, 255, 255),
                        font=font,
                    )
                except Exception as e:
                    print(f"Error adding alt position label: {e}")

                # Draw path to optimal position if requested
                if pathfinding and i < 3:  # Only for top 3
                    # Simple straight line path
                    draw.line(
                        [(opt_x, opt_y), (alt_x, alt_y)],
                        fill=(255, 255, 0),
                        width=1,
                    )

        # Add a text label with coordinates
        try:
            font = ImageFont.load_default()
            label_x = min(self.width - 150, max(10, opt_x))
            label_y = min(self.height - 50, max(10, opt_y - 30))
            draw.rectangle(
                [(label_x - 5, label_y - 5), (label_x + 145, label_y + 45)],
                fill=(0, 0, 0),
            )
            draw.text(
                (label_x, label_y),
                f"Optimal: ({opt_x}, {opt_y})",
                fill=(255, 255, 255),
                font=font,
            )
            draw.text(
                (label_x, label_y + 15),
                f"Visible spawns: {len(visible_spawns)}",
                fill=(255, 255, 255),
                font=font,
            )
            draw.text(
                (label_x, label_y + 30),
                f"Coverage: {len(visible_spawns)/self.total_spawns*100:.1f}%",
                fill=(255, 255, 255),
                font=font,
            )
        except Exception as e:
            print(f"Error adding text label: {e}")

        # Save the result
        output_path = "highlighted_spawn_tiles.png"
        image.save(output_path)
        print(f"Saved highlighted tile map to {output_path}")

        # Create a magnified version focused on the optimal position
        self._create_magnified_view(image, opt_x, opt_y, view_radius)

        return image

    def _create_magnified_view(self, image, center_x, center_y, view_radius):
        """
        Create a magnified view centered on the optimal position.

        :param image: Original PIL Image
        :param center_x: X-coordinate of center position
        :param center_y: Y-coordinate of center position
        :param view_radius: View radius
        """
        print("Creating magnified view of optimal position...")

        # Determine crop area
        magnify_radius = view_radius * 1.5
        left = max(0, center_x - magnify_radius)
        top = max(0, center_y - magnify_radius)
