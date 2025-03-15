import json
import argparse
import pandas as pd
import numpy as np
import os


def calculate_region_coords(world_x, world_y, plane=1):
    """
    Calculate regionId, regionX, and regionY from world coordinates.

    :param world_x: World X coordinate
    :param world_y: World Y coordinate
    :param plane: Game plane (default: 1)
    :return: Dictionary with regionId, regionX, regionY, and z
    """
    # Calculate region coordinates
    # OSRS uses 64x64 tile regions
    region_id = ((world_x >> 6) << 8) | (world_y >> 6)
    region_x = world_x & 0x3F  # world_x % 64
    region_y = world_y & 0x3F  # world_y % 64

    return {"regionId": region_id, "regionX": region_x, "regionY": region_y, "z": plane}


def generate_ground_markers(
    x_min, x_max, y_min, y_max, plane=1, color="#FFF100FF", csv_file=None
):
    """
    Generate ground markers for all tiles in the specified area.

    :param x_min: Minimum X coordinate
    :param x_max: Maximum X coordinate
    :param y_min: Minimum Y coordinate
    :param y_max: Maximum Y coordinate
    :param plane: Game plane (default: 1)
    :param color: Marker color (default: yellow - using RuneLite's default)
    :param csv_file: Optional CSV file with tiles to mark (1 = mark, 0 = skip)
    :return: List of ground marker objects
    """
    markers = []

    # If CSV file is provided, load it to determine which tiles to mark
    tile_map = None
    if csv_file and os.path.exists(csv_file):
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file)

            # Drop the first column (Y coordinates)
            tile_map = df.iloc[:, 1:].values

            print(f"Loaded tile map from {csv_file} with shape {tile_map.shape}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            print("Marking all tiles instead")

    # Iterate through all coordinates in the area
    for y in range(y_max, y_min - 1, -1):  # Iterate y in descending order
        for x in range(x_min, x_max + 1):
            # If we have a tile map, check if this tile should be marked
            if tile_map is not None:
                # Convert world coordinates to array indices
                row_idx = y_max - y
                col_idx = x - x_min

                # Skip if indices are out of bounds
                if (
                    row_idx < 0
                    or row_idx >= tile_map.shape[0]
                    or col_idx < 0
                    or col_idx >= tile_map.shape[1]
                ):
                    continue

                # Skip if tile is marked as 0 in the map
                if tile_map[row_idx, col_idx] == 0:
                    continue

            # Calculate region coordinates
            region_data = calculate_region_coords(x, y, plane)

            # Create marker with coordinates as label
            # Using exact format from the example provided
            marker = {
                "regionId": region_data["regionId"],
                "regionX": region_data["regionX"],
                "regionY": region_data["regionY"],
                "z": region_data["z"],
                "color": color,
                "label": f"{x},{y}",
            }

            markers.append(marker)

    return markers


def save_markers_to_json(markers, output_file="ground_markers.json"):
    """
    Save ground markers to a JSON file in exactly the format RuneLite expects.

    :param markers: List of ground marker objects
    :param output_file: Output JSON file name
    """
    try:
        # The format needs to be a plain JSON array with no whitespace for RuneLite to parse it correctly
        with open(output_file, "w") as f:
            json_str = json.dumps(markers, separators=(",", ":"))
            f.write(json_str)

        print(f"Saved {len(markers)} ground markers to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")

        # Create a backup with readable formatting for debugging
        debug_file = output_file.replace(".json", "_readable.json")
        with open(debug_file, "w") as f:
            json.dump(markers, f, indent=2)
        print(f"Also saved readable version to {debug_file} for debugging")
    except Exception as e:
        print(f"Error saving markers to JSON: {e}")


def generate_color_coded_markers(
    x_min, x_max, y_min, y_max, plane=1, walkable_csv=None, spawn_csv=None
):
    """
    Generate color-coded ground markers:
    - Yellow (#FFF100FF): Regular walkable tiles - RuneLite's default yellow
    - Green (#00FF00FF): Spawn locations
    - Purple (#FF00FFE7): Special/important tiles

    :param x_min: Minimum X coordinate
    :param x_max: Maximum X coordinate
    :param y_min: Minimum Y coordinate
    :param y_max: Maximum Y coordinate
    :param plane: Game plane
    :param walkable_csv: CSV file with walkable tiles
    :param spawn_csv: CSV file with spawn locations
    :return: List of all markers with appropriate colors
    """
    all_markers = []

    # Generate spawn location markers first (green)
    if spawn_csv and os.path.exists(spawn_csv):
        spawn_markers = generate_ground_markers(
            x_min, x_max, y_min, y_max, plane, "#00FF00FF", spawn_csv
        )
        all_markers.extend(spawn_markers)
        print(f"Generated {len(spawn_markers)} spawn location markers (green)")

    # Generate walkable tile markers (yellow)
    if walkable_csv and os.path.exists(walkable_csv):
        walkable_markers = generate_ground_markers(
            x_min, x_max, y_min, y_max, plane, "#FFF100FF", walkable_csv
        )
        # Filter out any duplicates that might already be in spawn markers
        existing_coords = set(
            f"{m['regionId']}:{m['regionX']}:{m['regionY']}" for m in all_markers
        )

        new_walkable_markers = [
            m
            for m in walkable_markers
            if f"{m['regionId']}:{m['regionX']}:{m['regionY']}" not in existing_coords
        ]

        all_markers.extend(new_walkable_markers)
        print(f"Generated {len(new_walkable_markers)} walkable tile markers (yellow)")

    # If no CSVs provided, mark all tiles
    if not (walkable_csv or spawn_csv):
        all_markers = generate_ground_markers(
            x_min, x_max, y_min, y_max, plane, "#FFF100FF"
        )
        print(f"Generated {len(all_markers)} markers for all tiles")

    return all_markers


def main():
    parser = argparse.ArgumentParser(
        description="Generate OSRS ground markers with coordinates as labels"
    )
    parser.add_argument("--x-min", type=int, required=True, help="Minimum X coordinate")
    parser.add_argument("--x-max", type=int, required=True, help="Maximum X coordinate")
    parser.add_argument("--y-min", type=int, required=True, help="Minimum Y coordinate")
    parser.add_argument("--y-max", type=int, required=True, help="Maximum Y coordinate")
    parser.add_argument("--plane", type=int, default=1, help="Game plane (default: 1)")
    parser.add_argument(
        "--color",
        default="#FFF100FF",
        help="Marker color in hex format with alpha (default: RuneLite yellow)",
    )
    parser.add_argument(
        "--output", default="ground_markers.json", help="Output JSON file name"
    )
    parser.add_argument(
        "--walkable",
        help="CSV file with walkable tiles (1 = walkable, 0 = not walkable)",
    )
    parser.add_argument(
        "--spawns", help="CSV file with spawn locations (1 = spawn, 0 = no spawn)"
    )
    parser.add_argument(
        "--optimal", help="Mark a specific optimal position (format: x,y)"
    )

    args = parser.parse_args()

    # If optimal position is specified, create a special marker for it
    if args.optimal:
        try:
            opt_x, opt_y = map(int, args.optimal.split(","))
            # Create a single marker for the optimal position
            region_data = calculate_region_coords(opt_x, opt_y, args.plane)
            optimal_marker = [
                {
                    "regionId": region_data["regionId"],
                    "regionX": region_data["regionX"],
                    "regionY": region_data["regionY"],
                    "z": region_data["z"],
                    "color": "#FF00FFE7",  # Purple
                    "label": f"OPTIMAL ({opt_x},{opt_y})",
                }
            ]
            save_markers_to_json(
                optimal_marker, f"optimal_position_{opt_x}_{opt_y}.json"
            )
            print(f"Created marker for optimal position at {opt_x},{opt_y}")
            return
        except:
            return

    if args.walkable or args.spawns:
        # Generate color-coded markers based on walkable tiles and spawn locations
        markers = generate_color_coded_markers(
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            args.plane,
            args.walkable,
            args.spawns,
        )
    else:
        # Generate markers for all tiles with the specified color
        markers = generate_ground_markers(
            args.x_min, args.x_max, args.y_min, args.y_max, args.plane, args.color
        )

    save_markers_to_json(markers, args.output)

    print("\nTo import these markers into RuneLite:")
    print("1. Open RuneLite")
    print("2. Right-click the Ground Markers plugin icon")
    print("3. Select 'Import'")
    print("4. Navigate to and select your generated JSON file")
    print("5. The markers will appear in-game with coordinate labels")


if __name__ == "__main__":
    main()

# Example usage:
# python ground_marker_generator.py --x-min 3718 --x-max 3761 --y-min 10248 --y-max 10295 --plane 1
#
# With CSV files:
# python ground_marker_generator.py --x-min 3718 --x-max 3761 --y-min 10248 --y-max 10295 --walkable walkable_tiles.csv --spawns spawn_locations.csv
