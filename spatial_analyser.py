import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import argparse
import time
import sys
import os


class SpawnOptimizer:
    def __init__(self, map_data):
        """
        Initialize the optimizer with a 2D map of possible spawn locations.

        :param map_data: 2D numpy array where 1 represents a valid spawn location, 0 represents invalid
        """
        self.map = np.array(map_data)
        self.height, self.width = self.map.shape

        print(f"Map dimensions: {self.width}x{self.height}")
        print(f"Total spawn locations: {np.sum(self.map)}")

    def find_optimal_position(self, view_radius=10, avoid_edges=True):
        """
        Find the optimal standing position that maximizes visible spawn locations.

        :param view_radius: Maximum radius the player can see from their position (default: 10)
        :param avoid_edges: If True, avoid positions near the edge of the map
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
        positions_checked = 0
        last_percent = 0
        start_time = time.time()

        print("\nCalculating optimal position...")
        print("[          ] 0% complete", end="\r")

        # Iterate through all positions on the map
        for y in range(self.height):
            for x in range(self.width):
                # Skip positions near the edge if requested
                if avoid_edges and not edge_mask[y, x]:
                    coverage_map[y, x] = -1  # Mark as invalid
                    continue

                # Count visible spawn locations from this position
                coverage_map[y, x] = self._count_visible_spawns(x, y, view_radius)

                # Update progress
                positions_checked += 1
                percent_complete = int((positions_checked / valid_positions) * 100)

                # Only update display when percentage changes to reduce console spam
                if percent_complete > last_percent:
                    last_percent = percent_complete
                    bars = int(percent_complete / 10)
                    progress_bar = "[" + "#" * bars + " " * (10 - bars) + "]"
                    elapsed = time.time() - start_time
                    positions_per_second = positions_checked / max(1, elapsed)
                    remaining_time = (valid_positions - positions_checked) / max(
                        1, positions_per_second
                    )

                    # Format time as h:m:s
                    remaining_min, remaining_sec = divmod(remaining_time, 60)
                    remaining_hour, remaining_min = divmod(remaining_min, 60)
                    time_str = f"{int(remaining_hour):02d}:{int(remaining_min):02d}:{int(remaining_sec):02d}"

                    print(
                        f"{progress_bar} {percent_complete}% complete - ETA: {time_str}",
                        end="\r",
                    )
                    sys.stdout.flush()

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
            return self.find_optimal_position(view_radius, avoid_edges=False)
        return None, None, 0

    def _count_visible_spawns(self, x, y, radius):
        """
        Count the number of spawn locations visible within a given radius.

        :param x: x-coordinate of the standing position
        :param y: y-coordinate of the standing position
        :param radius: View radius
        :return: Number of visible spawn locations
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

    def visualize_coverage(self, optimal_position=None, view_radius=10):
        """
        Create a heatmap visualization of the spawn coverage.

        :param optimal_position: Optional tuple of (x, y) for marking optimal position
        :param view_radius: View radius
        """
        print("\nGenerating coverage heatmap...")
        start_time = time.time()

        plt.figure(figsize=(15, 12))

        # Create coverage map
        coverage_map = np.zeros_like(self.map, dtype=int)

        # Use smaller step size for large maps to speed up visualization
        step = max(1, min(5, self.width // 200))
        print(f"Using step size of {step} for visualization sampling")

        # Calculate total positions for progress reporting
        total_positions = (self.height // step) * (self.width // step)
        positions_checked = 0
        last_percent = 0

        print("[          ] 0% complete", end="\r")

        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                coverage_map[y, x] = self._count_visible_spawns(x, y, view_radius)

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

        # Create heatmap of the coverage
        print("Drawing heatmap...")
        ax = sns.heatmap(
            coverage_map,
            cmap="YlOrRd",  # Yellow-Orange-Red colormap
            cbar=True,
            square=True,
            annot=False,
            fmt="d",
            linewidths=0.5,
        )

        # Mark optimal position if provided
        if optimal_position:
            x, y = optimal_position
            plt.plot(x + 0.5, y + 0.5, "b*", markersize=15)
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

    def visualize_map_with_optimal(self, optimal_position=None, view_radius=10):
        """
        Visualize the map with spawn locations and optimal position marked.

        :param optimal_position: Optional tuple of (x, y) for marking optimal position
        :param view_radius: View radius
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
        self, optimal_position, view_radius=10, original_image_path=None
    ):
        """
        Create a map with clear highlighting of each potential spawn tile and the optimal position.

        :param optimal_position: Tuple of (x, y) for the optimal position
        :param view_radius: View radius
        :param original_image_path: Path to the original image to overlay on
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

        # Add a text label with coordinates
        try:
            font = ImageFont.load_default()
            label_x = min(self.width - 150, max(10, opt_x))
            label_y = min(self.height - 50, max(10, opt_y - 30))
            draw.rectangle(
                [(label_x - 5, label_y - 5), (label_x + 145, label_y + 25)],
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
        right = min(self.width, center_x + magnify_radius)
        bottom = min(self.height, center_y + magnify_radius)

        # Crop the image
        cropped = image.crop((left, top, right, bottom))

        # Resize to make it larger (2x magnification)
        magnified = cropped.resize(
            (int(cropped.width * 2), int(cropped.height * 2)), Image.NEAREST
        )

        # Save the magnified view
        magnified_path = "magnified_optimal_position.png"
        magnified.save(magnified_path)
        print(f"Saved magnified view to {magnified_path}")

    def get_analysis(self, view_radius=10):
        """
        Provide detailed analysis of spawn distribution.

        :param view_radius: View radius
        :return: Dictionary of analysis metrics
        """
        # Find optimal position
        optimal_x, optimal_y, max_coverage = self.find_optimal_position(view_radius)

        # Count total spawn locations
        total_spawns = np.sum(self.map)

        # Calculate percentage of visible spawns
        visible_percent = 0
        if total_spawns > 0 and max_coverage > 0:
            visible_percent = (max_coverage / total_spawns) * 100

        return {
            "total_spawn_locations": int(total_spawns),
            "optimal_position": (optimal_x, optimal_y),
            "visible_spawns_count": max_coverage,
            "percent_of_total_visible": f"{visible_percent:.2f}%",
            "map_dimensions": f"{self.width}x{self.height}",
            "view_radius": view_radius,
        }


def import_spawn_map_from_image(image_path, save_debug_image=True):
    """
    Import a spawn map from an image, identifying red squares as spawn locations.

    :param image_path: Path to the image file
    :param save_debug_image: Whether to save a debug image showing detected spawn points
    :return: 2D numpy array with binary spawn location map
    """
    try:
        print(f"Loading image: {image_path}")
        start_time = time.time()

        # Open the image
        img = Image.open(image_path)
        print(f"Successfully opened image with size {img.size}")

        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert image to numpy array
        img_array = np.array(img)
        print("Processing image to identify red squares...")

        # Create a binary mask for pure red color (with stricter tolerance)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # More precise detection of bright red squares
        # These are the specific thresholds for detecting pure red squares
        red_threshold = 200  # Very high red value
        other_threshold = 50  # Very low green and blue values
        red_dominance = 150  # How much brighter red should be than other channels

        red_mask = (
            (r > red_threshold)  # High red
            & (g < other_threshold)  # Low green
            & (b < other_threshold)  # Low blue
            & (r - g > red_dominance)  # Red much brighter than green
            & (r - b > red_dominance)  # Red much brighter than blue
        )

        # Try to clean up the mask with morphological operations
        try:
            from scipy import ndimage

            print("Applying morphological operations to clean up detection...")
            red_mask = ndimage.binary_erosion(red_mask, iterations=1)
            red_mask = ndimage.binary_dilation(red_mask, iterations=1)
        except ImportError:
            print("SciPy not available, skipping morphological cleanup")

        # Convert to integer array (0 or 1)
        spawn_map = red_mask.astype(int)

        # Optional: Save a debug image showing detected spawn points
        if save_debug_image:
            print("Saving debug image to visualize detection...")
            debug_img = img_array.copy()
            # Mark detected spawn locations in green
            debug_img[red_mask] = [0, 255, 0]
            debug_pil = Image.fromarray(debug_img)
            debug_path = "debug_detected_spawns.png"
            debug_pil.save(debug_path)
            print(f"Saved debug image to {debug_path}")

        # Count spawns and downsample if needed
        spawn_count = np.sum(spawn_map)
        print(f"Found {spawn_count} spawn locations (red pixels)")

        if (
            spawn_count > 500
        ):  # If we detect too many spawns, apply grid-based detection
            print("Detected many red pixels, applying grid-based analysis...")
            # Try a grid-based approach to find the center of red squares
            grid_spawn_map = np.zeros_like(spawn_map)
            height, width = spawn_map.shape

            # Estimate grid size based on image dimensions (adjust as needed)
            grid_size = 20
            print(f"Using grid size of {grid_size}x{grid_size} pixels")
            print("Processing grid cells...")

            total_cells = ((height + grid_size - 1) // grid_size) * (
                (width + grid_size - 1) // grid_size
            )
            cells_processed = 0
            last_percent = 0

            print("[          ] 0% complete", end="\r")

            for y in range(0, height, grid_size):
                for x in range(0, width, grid_size):
                    # Check a grid cell
                    end_y = min(y + grid_size, height)
                    end_x = min(x + grid_size, width)

                    cell = spawn_map[y:end_y, x:end_x]
                    # If significant portion of cell is red, mark as spawn
                    if np.sum(cell) > (grid_size * grid_size * 0.3):  # 30% threshold
                        center_y = y + (end_y - y) // 2
                        center_x = x + (end_x - x) // 2
                        grid_spawn_map[center_y, center_x] = 1

                    # Update progress
                    cells_processed += 1
                    percent_complete = int((cells_processed / total_cells) * 100)

                    if percent_complete > last_percent:
                        last_percent = percent_complete
                        bars = int(percent_complete / 10)
                        progress_bar = "[" + "#" * bars + " " * (10 - bars) + "]"
                        print(f"{progress_bar} {percent_complete}% complete", end="\r")
                        sys.stdout.flush()

            # Clear the progress line
            print(" " * 80, end="\r")

            spawn_map = grid_spawn_map
            print(
                f"After grid processing: {np.sum(spawn_map)} distinct spawn locations"
            )

            # Save another debug image with grid-based spawn points
            if save_debug_image:
                print("Saving grid-based debug image...")
                grid_debug_img = img_array.copy()
                for y in range(grid_spawn_map.shape[0]):
                    for x in range(grid_spawn_map.shape[1]):
                        if grid_spawn_map[y, x] == 1:
                            # Draw a green cross at each spawn point
                            size = 5
                            y1, y2 = max(0, y - size), min(
                                grid_debug_img.shape[0] - 1, y + size
                            )
                            x1, x2 = max(0, x - size), min(
                                grid_debug_img.shape[1] - 1, x + size
                            )

                            # Horizontal line
                            grid_debug_img[y, x1 : x2 + 1] = [0, 255, 0]
                            # Vertical line
                            grid_debug_img[y1 : y2 + 1, x] = [0, 255, 0]

                grid_debug_pil = Image.fromarray(grid_debug_img)
                grid_debug_path = "grid_spawn_points.png"
                grid_debug_pil.save(grid_debug_path)
                print(f"Saved grid-based debug image to {grid_debug_path}")

        elapsed = time.time() - start_time
        print(f"Map processing complete in {elapsed:.2f} seconds!")
        return spawn_map

    except Exception as e:
        print(f"Error importing image: {e}")
        return None


def downsample_map(spawn_map, factor=4):
    """
    Downsample the map to reduce computation time.

    :param spawn_map: Original spawn map
    :param factor: Downsample factor
    :return: Downsampled map
    """
    print(f"Downsampling map by factor of {factor}...")
    height, width = spawn_map.shape
    new_height, new_width = height // factor, width // factor

    downsampled = np.zeros((new_height, new_width), dtype=int)

    for y in range(new_height):
        for x in range(new_width):
            # Check if any pixels in this block are spawn locations
            if np.any(
                spawn_map[y * factor : (y + 1) * factor, x * factor : (x + 1) * factor]
            ):
                downsampled[y, x] = 1

    print(f"Downsampled from {width}x{height} to {new_width}x{new_height}")
    print(f"Original spawn locations: {np.sum(spawn_map)}")
    print(f"Downsampled spawn locations: {np.sum(downsampled)}")

    return downsampled


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Find optimal position for maximum spawn visibility"
    )
    parser.add_argument(
        "--image", type=str, default="spawn_map.png", help="Path to map image"
    )
    parser.add_argument("--radius", type=int, default=10, help="View radius")
    parser.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="Downsample factor (0 for no downsampling)",
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable debug image generation"
    )
    parser.add_argument(
        "--no-edge-avoid", action="store_true", help="Disable edge avoidance"
    )
    args = parser.parse_args()

    print("\n=== Spawn Location Optimizer ===\n")
    start_time = time.time()

    # Import map from image
    image_path = args.image
    spawn_map = import_spawn_map_from_image(
        image_path, save_debug_image=not args.no_debug
    )

    if spawn_map is None:
        print("Failed to load image. Please check the file path.")
        return

    # Downsample the map if requested
    # Downsample the map if requested
    if args.downsample > 1:
        spawn_map = downsample_map(spawn_map, args.downsample)

    # Set the view radius
    view_radius = args.radius
    print(f"Using view radius of {view_radius} tiles")

    # Create optimizer
    optimizer = SpawnOptimizer(spawn_map)

    # Find optimal position
    optimal_x, optimal_y, coverage = optimizer.find_optimal_position(
        view_radius, avoid_edges=not args.no_edge_avoid
    )

    if optimal_x is not None:
        print(f"Optimal Position: ({optimal_x}, {optimal_y})")
        print(f"Visible spawn locations: {coverage}")

        # Create a special highlighted tile map
        optimizer.create_highlighted_tile_map(
            (optimal_x, optimal_y), view_radius, image_path
        )

        # Visualize the map with optimal position
        optimizer.visualize_map_with_optimal((optimal_x, optimal_y), view_radius)

        # Visualize coverage heatmap
        optimizer.visualize_coverage((optimal_x, optimal_y), view_radius)

        # Get detailed analysis
        analysis = optimizer.get_analysis(view_radius)
        print("\nSpawn Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")

        # Generate a text file with the results
        with open("optimal_position_results.txt", "w") as f:
            f.write("=== Spawn Location Optimizer Results ===\n\n")
            f.write(f"Optimal Position: ({optimal_x}, {optimal_y})\n")
            f.write(f"Visible spawn locations: {coverage}\n\n")
            f.write("Spawn Analysis:\n")
            for key, value in analysis.items():
                f.write(f"{key}: {value}\n")

        print("\nResults saved to optimal_position_results.txt")
    else:
        print("No valid positions found!")

    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    print("Done!")


if __name__ == "__main__":
    main()
