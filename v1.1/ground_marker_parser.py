import json
import csv
import os

def parse_ground_markers(json_data, output_csv=None, start_marker_label="START"):
    """
    Parse ground markers JSON data and convert to CSV format for spawn testing
    
    :param json_data: JSON string or dict with ground marker data
    :param output_csv: Path to output CSV file (optional)
    :param start_marker_label: Label that marks the player's starting position
    :return: tuple of (start_position, spawn_data) where spawn_data is a list of (x, y, count) tuples
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        markers = json.loads(json_data)
    else:
        markers = json_data
    
    # Find the starting position (player's position)
    start_position = None
    for marker in markers:
        if marker.get("label") == start_marker_label:
            start_position = (marker["regionX"], marker["regionY"])
            break
    
    if not start_position:
        print(f"Warning: No marker with label '{start_marker_label}' found. Using the first marker as start position.")
        start_position = (markers[0]["regionX"], markers[0]["regionY"])
    
    # Collect spawn points - assume they have a numerical label for count
    # If label is missing or non-numeric, assume count=1
    spawn_data = []
    
    for marker in markers:
        # Skip the start marker
        if marker.get("label") == start_marker_label:
            continue
        
        x, y = marker["regionX"], marker["regionY"]
        
        # Get the count from the label if present
        count = 1
        if "label" in marker:
            try:
                count = int(marker["label"])
            except ValueError:
                # If label isn't a number, default to 1
                count = 1
        
        spawn_data.append((x, y, count))
    
    # If output CSV path is provided, write the data
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["x", "y", "count"])
            # Write spawn data
            for x, y, count in spawn_data:
                writer.writerow([x, y, count])
        
        print(f"Converted {len(spawn_data)} spawn markers to CSV file: {output_csv}")
    
    # Calculate total observations
    total_observations = sum(count for _, _, count in spawn_data)
    print(f"Found {len(spawn_data)} unique spawn locations with {total_observations} total observations")
    print(f"Player starting position at ({start_position[0]}, {start_position[1]})")
    
    return start_position, spawn_data

def process_ground_markers_file(input_file, output_csv=None):
    """
    Process a file containing ground marker JSON data
    
    :param input_file: Path to JSON file with ground marker data
    :param output_csv: Path to output CSV file (optional)
    :return: tuple of (start_position, spawn_data)
    """
    try:
        with open(input_file, 'r') as f:
            json_data = f.read()
        
        return parse_ground_markers(json_data, output_csv)
    
    except Exception as e:
        print(f"Error processing ground markers file: {e}")
        return None, []

if __name__ == "__main__":
    # Simple command-line interface if run directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert RuneScape ground markers to spawn data CSV')
    parser.add_argument('input_file', help='JSON file containing ground markers')
    parser.add_argument('--output', '-o', help='Output CSV file (default: spawns.csv)', default='spawns.csv')
    parser.add_argument('--start-label', '-s', help='Label for the starting position marker (default: START)', 
                        default='START')
    
    args = parser.parse_args()
    
    start_pos, spawn_data = process_ground_markers_file(args.input_file, args.output)
    
    if start_pos:
        print(f"\nConversion complete! You can now use this data with the spawn testing module:")
        print(f"python main.py --test-spawns --spawn-data {args.output} --test-pos-x {start_pos[0]} --test-pos-y {start_pos[1]}")