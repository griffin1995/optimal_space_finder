import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from spatial_analyser import convert_to_game_coordinates


def calculate_best_positions_by_sector(
    self, view_radius=10, sectors=4, use_tile_distance=False
):
    """
    Find best positions by dividing the map into sectors

    :param view_radius: View radius or tile distance
    :param sectors: Number of sectors (square root of total sectors)
    :param use_tile_distance: Whether to use tile distance instead of Euclidean
    :return: List of best positions per sector with coverage
    """
    distance_type = "tile distance" if use_tile_distance else "view radius"
    print(
        f"Analyzing map by dividing into {sectors}x{sectors} sectors using {distance_type}..."
    )

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
                    # Only check valid positions (not too close to sector edge and walkable)
                    if (
                        x >= x_start + view_radius // 2
                        and x < x_end - view_radius // 2
                        and y >= y_start + view_radius // 2
                        and y < y_end - view_radius // 2
                        and self.walkable_map[y, x] > 0
                    ):
                        if use_tile_distance:
                            coverage = self._count_visible_spawns_tile_distance(
                                x, y, view_radius
                            )
                        else:
                            coverage = self._count_visible_spawns_optimized(
                                x, y, view_radius
                            )

                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_pos = (x, y)

            if best_pos:
                game_x, game_y = convert_to_game_coordinates(*best_pos)
                best_positions.append((best_pos[0], best_pos[1], best_coverage))
                print(
                    f"Best position in sector ({sx},{sy}): {best_pos} (Game: {game_x}, {game_y}) "
                    f"with {best_coverage} visible spawns"
                )

    # Sort by coverage
    best_positions.sort(key=lambda x: x[2], reverse=True)
    return best_positions


def create_combined_optimal_locations_graph(results, optimizer):
    """
    Create a graph showing all optimal locations for different radii on the same map.

    :param results: List of tuples (radius, opt_x, opt_y, game_x, game_y, coverage, percentage)
    :param optimizer: SpawnOptimizer instance
    """
    print("\nCreating combined optimal locations graph...")

    fig, ax = plt.subplots(figsize=(15, 12))

    # Create a background map
    visualization = np.zeros((optimizer.height, optimizer.width, 3), dtype=np.uint8)

    # Set spawn locations to light red
    mask = optimizer.map == 1
    visualization[mask] = [180, 0, 0]  # Red for spawn locations

    # Set walkable tiles to dark gray
    walkable_mask = (optimizer.walkable_map == 1) & ~mask
    visualization[walkable_mask] = [50, 50, 50]  # Dark gray for walkable tiles

    ax.imshow(visualization)

    # Create a colormap for radius values
    # If there's only one radius, we need to handle it differently
    if len(results) == 1:
        # Use a fixed color for a single radius
        radius = results[0][0]
        opt_x, opt_y = results[0][1], results[0][2]
        game_x, game_y = results[0][3], results[0][4]
        coverage = results[0][5]

        # Plot with a fixed color for single radius
        marker_size = 150
        ax.scatter(
            opt_x, opt_y, s=marker_size, color="blue", alpha=0.7, edgecolors="white"
        )

        # Add a text label with radius, position, and game coordinates
        ax.annotate(
            f"R{radius}: ({opt_x},{opt_y})\nGame: ({game_x},{game_y})",
            (opt_x, opt_y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color="white",
            backgroundcolor="black",
        )

        # No need for colorbar with a single radius
    else:
        # For multiple radii, use a colormap
        min_radius = min(r[0] for r in results)
        max_radius = max(r[0] for r in results)
        norm = colors.Normalize(vmin=min_radius, vmax=max_radius)
        cmap = cm.viridis

        # Plot each optimal position with a color based on radius
        for radius, opt_x, opt_y, game_x, game_y, coverage, percentage in results:
            color = cmap(norm(radius))
            # Plot with larger marker size for larger radii
            marker_size = 100 + (radius - min_radius) * 10
            ax.scatter(
                opt_x, opt_y, s=marker_size, color=color, alpha=0.7, edgecolors="white"
            )

            # Add a text label with radius, position, and game coordinates
            ax.annotate(
                f"R{radius}: ({opt_x},{opt_y})\nGame: ({game_x},{game_y})",
                (opt_x, opt_y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color="white",
                backgroundcolor="black",
            )

        # Create a colorbar to show the radius scale
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("View Radius/Tile Distance")

    ax.set_title("Optimal Positions for Different View Radii/Tile Distances")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, color="gray", linestyle="--", alpha=0.3)

    # Save the visualization
    plt.tight_layout()
    plt.savefig("combined_optimal_positions.png")
    print("Combined optimal positions graph saved to combined_optimal_positions.png")

    # Also create a zoomed version showing just the region of interest
    create_zoomed_optimal_locations_graph(results, optimizer)

    return plt


def create_zoomed_optimal_locations_graph(results, optimizer):
    """
    Create a zoomed version of the combined graph focusing on the region with optimal positions.

    :param results: List of tuples (radius, opt_x, opt_y, game_x, game_y, coverage, percentage)
    :param optimizer: SpawnOptimizer instance
    """
    # Get the bounds of the optimal positions
    min_x = min(r[1] for r in results)
    max_x = max(r[1] for r in results)
    min_y = min(r[2] for r in results)
    max_y = max(r[2] for r in results)

    # Add padding
    padding = max(20, max(r[0] for r in results))
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(optimizer.width, max_x + padding)
    max_y = min(optimizer.height, max_y + padding)

    # Create the zoomed graph
    fig, ax = plt.subplots(figsize=(15, 12))

    # Create a cropped background map
    visualization = np.zeros((optimizer.height, optimizer.width, 3), dtype=np.uint8)
    mask = optimizer.map == 1
    visualization[mask] = [180, 0, 0]  # Red for spawn locations
    walkable_mask = (optimizer.walkable_map == 1) & ~mask
    visualization[walkable_mask] = [50, 50, 50]  # Dark gray for walkable tiles

    # Cropped view
    cropped_vis = visualization[int(min_y) : int(max_y), int(min_x) : int(max_x)]
    ax.imshow(cropped_vis, extent=[min_x, max_x, max_y, min_y])

    # Handle differently for a single radius vs multiple radii
    if len(results) == 1:
        # Use a fixed color for a single radius
        radius = results[0][0]
        opt_x, opt_y = results[0][1], results[0][2]
        game_x, game_y = results[0][3], results[0][4]

        # Plot with a fixed color for single radius
        marker_size = 150
        ax.scatter(
            opt_x, opt_y, s=marker_size, color="blue", alpha=0.7, edgecolors="white"
        )

        # Add a text label with radius, position, and game coordinates
        ax.annotate(
            f"R{radius}: ({opt_x},{opt_y})\nGame: ({game_x},{game_y})",
            (opt_x, opt_y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color="white",
            backgroundcolor="black",
        )
    else:
        # For multiple radii, use a colormap
        min_radius = min(r[0] for r in results)
        max_radius = max(r[0] for r in results)
        norm = colors.Normalize(vmin=min_radius, vmax=max_radius)
        cmap = cm.viridis

        # Plot each optimal position
        for radius, opt_x, opt_y, game_x, game_y, coverage, percentage in results:
            color = cmap(norm(radius))
            marker_size = 100 + (radius - min_radius) * 10
            ax.scatter(
                opt_x, opt_y, s=marker_size, color=color, alpha=0.7, edgecolors="white"
            )

            ax.annotate(
                f"R{radius}: ({opt_x},{opt_y})\nGame: ({game_x},{game_y})",
                (opt_x, opt_y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color="white",
                backgroundcolor="black",
            )

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("View Radius/Tile Distance")

    ax.set_title("Zoomed View of Optimal Positions")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("zoomed_optimal_positions.png")
    print("Zoomed optimal positions graph saved to zoomed_optimal_positions.png")

    return plt
