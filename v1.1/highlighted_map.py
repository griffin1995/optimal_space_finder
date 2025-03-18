import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont
from spatial_analyser import convert_to_game_coordinates


def create_highlighted_tile_map(
    self,
    optimal_position,
    view_radius=None,
    max_tiles=None,
    original_image_path=None,
    alternative_positions=None,
    pathfinding=False,
    game_coordinates=None,
    use_tile_distance=False,
):
    """
    Create a map with clear highlighting of each potential spawn tile and the optimal position.

    :param optimal_position: Tuple of (x, y) for the optimal position
    :param view_radius: View radius (for Euclidean distance)
    :param max_tiles: Maximum tile distance (for tile distance)
    :param original_image_path: Path to the original image to overlay on
    :param alternative_positions: List of alternative positions to mark
    :param pathfinding: If True, show paths between optimal positions
    :param game_coordinates: Tuple of (game_x, game_y) for the optimal position
    :param use_tile_distance: Whether to use tile distance instead of Euclidean
    """
    if optimal_position is None or optimal_position[0] is None:
        print("No optimal position to mark!")
        return

    if view_radius is None and max_tiles is None:
        raise ValueError("Either view_radius or max_tiles must be provided")

    if use_tile_distance and max_tiles is None and view_radius is not None:
        max_tiles = view_radius  # Use view_radius as max_tiles if not provided

    distance_type = "tile distance" if use_tile_distance else "view radius"
    distance_value = max_tiles if use_tile_distance else view_radius

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

    if use_tile_distance and max_tiles is not None:
        # Find reachable tiles and collect visible spawns
        reachable = self._find_reachable_tiles_tile_distance(opt_x, opt_y, max_tiles)
        for spawn_y, spawn_x in self.spawn_locations:
            if (spawn_y, spawn_x) in reachable:
                visible_spawns.append((spawn_x, spawn_y))
    else:
        # Use Euclidean distance
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

    # Draw the view radius or reachable area
    print(f"Drawing {distance_type}...")
    if use_tile_distance and max_tiles is not None:
        # Draw reachable tiles as cyan dots
        reachable = self._find_reachable_tiles_tile_distance(opt_x, opt_y, max_tiles)
        for tile_y, tile_x in reachable:
            # Draw a small cyan pixel for each reachable tile
            if (tile_y, tile_x) != (opt_y, opt_x):  # Skip the central position
                draw.point((tile_x, tile_y), fill=(0, 255, 255))
    else:
        # Draw the view radius circle
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
        label_x = min(self.width - 200, max(10, opt_x))
        label_y = min(self.height - 70, max(10, opt_y - 30))
        draw.rectangle(
            [(label_x - 5, label_y - 5), (label_x + 195, label_y + 65)],
            fill=(0, 0, 0),
        )

        # Use game coordinates if provided
        if game_coordinates:
            game_x, game_y = game_coordinates
        else:
            game_x, game_y = convert_to_game_coordinates(opt_x, opt_y)

        draw.text(
            (label_x, label_y),
            f"Map Position: ({opt_x}, {opt_y})",
            fill=(255, 255, 255),
            font=font,
        )
        draw.text(
            (label_x, label_y + 15),
            f"Game Position: ({game_x}, {game_y})",
            fill=(255, 255, 255),
            font=font,
        )
        draw.text(
            (label_x, label_y + 30),
            f"Visible spawns: {len(visible_spawns)}",
            fill=(255, 255, 255),
            font=font,
        )
        draw.text(
            (label_x, label_y + 45),
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
    self._create_magnified_view(
        image,
        opt_x,
        opt_y,
        max(max_tiles if max_tiles else 0, view_radius if view_radius else 0),
    )

    return image


def _create_magnified_view(self, image, center_x, center_y, view_radius):
    """
    Create a magnified view centered on the optimal position.

    :param image: Original PIL Image
    :param center_x: X-coordinate of center position
    :param center_y: Y-coordinate of center position
    :param view_radius: View radius or maximum tile distance
    """
    print("Creating magnified view of optimal position...")

    # Determine crop area
    magnify_radius = view_radius * 1.5
    left = max(0, center_x - magnify_radius)
    top = max(0, center_y - magnify_radius)
    right = min(self.width - 1, center_x + magnify_radius)
    bottom = min(self.height - 1, center_y + magnify_radius)

    # Convert to integers
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)

    # Ensure we have a valid crop area
    if right <= left or bottom <= top:
        print("Invalid crop area. Skipping magnified view.")
        return

    # Crop the image
    cropped = image.crop((left, top, right, bottom))

    # Calculate scale factor (aim for a 800x800 image)
    target_size = 800
    width, height = cropped.size
    scale = min(target_size / width, target_size / height)

    # Scale up the cropped image
    if scale > 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        magnified = cropped.resize((new_width, new_height), Image.LANCZOS)
    else:
        magnified = cropped

    # Save the magnified view
    magnified_path = "magnified_optimal_position.png"
    magnified.save(magnified_path)
    print(f"Saved magnified view to {magnified_path}")
