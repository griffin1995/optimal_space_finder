import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
import os
from PIL import Image, ImageDraw, ImageFont
from spatial_analyser import convert_to_game_coordinates


def visualize_coverage(
    self,
    optimal_position=None,
    view_radius=None,
    max_tiles=None,
    sample_step=None,
    use_tile_distance=False,
):
    """
    Create a heatmap visualization of the spawn coverage.

    :param optimal_position: Optional tuple of (x, y) for marking optimal position
    :param view_radius: View radius (for Euclidean distance)
    :param max_tiles: Maximum tile distance (for tile distance)
    :param sample_step: Step size for sampling (None for automatic)
    :param use_tile_distance: Whether to use tile distance instead of Euclidean
    """
    print("\nGenerating coverage heatmap...")

    if view_radius is None and max_tiles is None:
        raise ValueError("Either view_radius or max_tiles must be provided")

    if use_tile_distance and max_tiles is None:
        max_tiles = view_radius  # Use view_radius as max_tiles if not provided

    distance_type = "tile distance" if use_tile_distance else "view radius"
    distance_value = max_tiles if use_tile_distance else view_radius

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
            # Skip unwalkable positions
            if not self.walkable_map[y, x]:
                continue

            # Count visible spawns using appropriate method
            if use_tile_distance:
                coverage_map[y, x] = self._count_visible_spawns_tile_distance(
                    x, y, max_tiles
                )
            else:
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
    plt.scatter(spawn_x + 0.5, spawn_y + 0.5, color="red", s=10, alpha=0.5, marker="o")

    # Mark optimal position if provided
    if optimal_position:
        x, y = optimal_position
        plt.plot(x + 0.5, y + 0.5, "r*", markersize=15)
        game_x, game_y = convert_to_game_coordinates(x, y)
        plt.title(
            f"Spawn Coverage Map ({distance_type.capitalize()} {distance_value})\n"
            f"Optimal Position: ({x},{y}) - Game: ({game_x},{game_y})\n"
            f"Visible Spawns: {coverage_map[y,x]}"
        )
    else:
        plt.title(f"Spawn Coverage Map ({distance_type.capitalize()} {distance_value})")

    plt.tight_layout()

    # Save the visualization
    heatmap_path = "optimal_position_heatmap.png"
    plt.savefig(heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")

    elapsed = time.time() - start_time
    print(f"Heatmap generated in {elapsed:.2f} seconds!")

    return coverage_map


def visualize_map_with_optimal(
    self,
    optimal_position=None,
    view_radius=None,
    max_tiles=None,
    mark_alternatives=False,
    alternative_positions=None,
    use_tile_distance=False,
):
    """
    Visualize the map with spawn locations and optimal position marked.

    :param optimal_position: Optional tuple of (x, y) for marking optimal position
    :param view_radius: View radius (for Euclidean distance)
    :param max_tiles: Maximum tile distance (for tile distance)
    :param mark_alternatives: If True, mark alternative good positions
    :param alternative_positions: List of alternative positions to mark
    :param use_tile_distance: Whether to use tile distance instead of Euclidean
    """
    if view_radius is None and max_tiles is None and optimal_position is not None:
        raise ValueError("Either view_radius or max_tiles must be provided")

    if use_tile_distance and max_tiles is None and view_radius is not None:
        max_tiles = view_radius  # Use view_radius as max_tiles if not provided

    distance_type = "tile distance" if use_tile_distance else "view radius"
    distance_value = max_tiles if use_tile_distance else view_radius

    print("\nGenerating map visualization...")
    start_time = time.time()

    plt.figure(figsize=(15, 12))

    # Create a colored visualization
    visualization = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    # Set spawn locations to red
    mask = self.map == 1
    visualization[mask] = [255, 0, 0]  # Red for spawn locations

    # Set walkable tiles to dark gray
    walkable_mask = (self.walkable_map == 1) & ~mask
    visualization[walkable_mask] = [50, 50, 50]  # Dark gray for walkable tiles

    plt.imshow(visualization)

    # Mark optimal position if provided
    if optimal_position:
        x, y = optimal_position
        plt.plot(x, y, "b*", markersize=15)

        # Convert to game coordinates
        game_x, game_y = convert_to_game_coordinates(x, y)

        plt.title(
            f"Spawn Map with Optimal Position ({distance_type.capitalize()} {distance_value})\n"
            f"Map: ({x},{y}) - Game: ({game_x},{game_y})"
        )

        # Draw the view radius or reachable tiles
        if use_tile_distance and max_tiles is not None:
            # Show reachable tiles as shaded area
            reachable = self._find_reachable_tiles_tile_distance(x, y, max_tiles)
            for tile_y, tile_x in reachable:
                if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
                    plt.plot(tile_x, tile_y, "co", markersize=2, alpha=0.3)
        elif view_radius is not None:
            # Draw the view radius circle
            circle = plt.Circle(
                (x, y), view_radius, color="b", fill=False, linestyle="--"
            )
            plt.gca().add_patch(circle)

        print("Calculating visible spawns from optimal position...")
        # Draw the coverage area - visible spawns
        visible_count = 0

        if use_tile_distance and max_tiles is not None:
            # Find reachable tiles and highlight visible spawns
            reachable = self._find_reachable_tiles_tile_distance(x, y, max_tiles)
            for spawn_y, spawn_x in self.spawn_locations:
                if (spawn_y, spawn_x) in reachable:
                    plt.plot(spawn_x, spawn_y, "go", markersize=5)
                    visible_count += 1
        else:
            # Use Euclidean distance
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
    return visualization
