import json
import numpy as np
import argparse
import os

def parse_ground_markers(json_file):
    """
    Parse a ground markers JSON file and extract tile coordinates
    
    :param json_file: Path to JSON file with ground markers
    :return: Set of (x, y) tuples representing tile coordinates
    """
    try:
        # Read the JSON file
        with open(json_file, 'r') as f:
            markers = json.load(f)
        
        # Extract coordinates from markers
        # We need to convert from region coordinates back to world coordinates
        tiles = set()
        for marker in markers:
            region_id = marker['regionId']
            region_x = marker['regionX']
            region_y = marker['regionY']
            
            # Convert region coordinates to world coordinates
            # RegionID encodes the top-level region coordinates
            region_x_base = (region_id >> 8) << 6
            region_y_base = (region_id & 0xFF) << 6
            
            # Calculate world coordinates
            world_x = region_x_base + region_x
            world_y = region_y_base + region_y
            
            tiles.add((world_x, world_y))
        
        return tiles
    except Exception as e:
        print(f"Error parsing ground markers file {json_file}: {e}")
        return set()

def generate_walkable_array(all_tiles, bad_tiles, x_min, x_max, y_min, y_max):
    """
    Generate a 2D array representing walkable tiles
    
    :param all_tiles: Set of all tile coordinates
    :param bad_tiles: Set of non-walkable tile coordinates
    :param x_min: Minimum X coordinate
    :param x_max: Maximum X coordinate
    :param y_min: Minimum Y coordinate
    :param y_max: Maximum Y coordinate
    :return: 2D numpy array with 1 for walkable, 0 for non-walkable
    """
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    # Initialize array with zeros
    walkable_map = np.zeros((height, width), dtype=int)
    
    # Determine walkable tiles (in all_tiles but not in bad_tiles)
    walkable_tiles = all_tiles - bad_tiles
    
    # Mark walkable tiles
    for x, y in walkable_tiles:
        # Convert world coordinates to array indices
        row = y_max - y  # Y is inverted in the array
        col = x - x_min
        
        # Check if within bounds
        if 0 <= row < height and 0 <= col < width:
            walkable_map[row, col] = 1
    
    return walkable_map

def export_array_to_txt(array, output_file):
    """
    Export a numpy array to a text file
    
    :param array: Numpy array to export
    :param output_file: Output file path
    """
    np.savetxt(output_file, array, fmt='%d')
    print(f"Exported array to {output_file}")
    print(f"Array shape: {array.shape}")
    print(f"Walkable tiles (1s): {np.sum(array)}")
    print(f"Non-walkable tiles (0s): {array.size - np.sum(array)}")

def export_array_to_csv(array, output_file, x_min, y_min):
    """
    Export a numpy array to a CSV file with coordinates
    
    :param array: Numpy array to export
    :param output_file: Output file path
    :param x_min: Minimum X coordinate for column headers
    :param y_min: Minimum Y coordinate for row headers
    """
    height, width = array.shape
    
    # Create CSV content
    with open(output_file, 'w') as f:
        # Write header row with X coordinates
        header = ['Y/X'] + [str(x_min + i) for i in range(width)]
        f.write(','.join(header) + '\n')
        
        # Write data rows with Y coordinates
        for row in range(height):
            y = y_min + (height - row - 1)  # Invert Y to match OSRS coordinates
            row_data = [str(y)] + [str(array[row, col]) for col in range(width)]
            f.write(','.join(row_data) + '\n')
    
    print(f"Exported CSV to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate walkable tiles array from ground marker JSON files')
    parser.add_argument('--all-tiles', required=True, help='JSON file with all tiles marked')
    parser.add_argument('--bad-tiles', required=True, help='JSON file with non-walkable tiles marked')
    parser.add_argument('--x-min', type=int, required=True, help='Minimum X coordinate')
    parser.add_argument('--x-max', type=int, required=True, help='Maximum X coordinate')
    parser.add_argument('--y-min', type=int, required=True, help='Minimum Y coordinate')
    parser.add_argument('--y-max', type=int, required=True, help='Maximum Y coordinate')
    parser.add_argument('--output-txt', default='walkable_tiles.txt', help='Output text file for array')
    parser.add_argument('--output-csv', default='walkable_tiles.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    print(f"Processing area: ({args.x_min},{args.y_min}) to ({args.x_max},{args.y_max})")
    
    # Parse ground marker files
    print(f"Parsing all tiles file: {args.all_tiles}")
    all_tiles = parse_ground_markers(args.all_tiles)
    print(f"Found {len(all_tiles)} tiles in all-tiles file")
    
    print(f"Parsing bad tiles file: {args.bad_tiles}")
    bad_tiles = parse_ground_markers(args.bad_tiles)
    print(f"Found {len(bad_tiles)} tiles in bad-tiles file")
    
    # Generate walkable array
    print("Generating walkable tiles array...")
    walkable_array = generate_walkable_array(
        all_tiles, bad_tiles, args.x_min, args.x_max, args.y_min, args.y_max
    )
    
    # Export array to files
    export_array_to_txt(walkable_array, args.output_txt)
    export_array_to_csv(walkable_array, args.output_csv, args.x_min, args.y_min)
    
    print("\nProcess completed successfully!")
    print(f"Generated walkable tiles array with {np.sum(walkable_array)} walkable tiles")
    print(f"Walkable percentage: {np.sum(walkable_array) / walkable_array.size * 100:.2f}%")

if __name__ == '__main__':
    main()