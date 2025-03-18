# Spawn Optimizer

This tool helps find optimal positions to view the maximum number of spawn locations on a map, with specific application for seaweed spore collection in Old School RuneScape.

## Files

- `spatial_analyser.py` - Core class definition
- `euclidean_distance.py` - Euclidean distance pathfinding methods
- `tile_distance.py` - Tile distance pathfinding methods
- `visualization.py` - Map and heatmap visualization functions
- `highlighted_map.py` - Highlighted tile map creation functions
- `other_functions.py` - Additional helper functions
- `spawn_testing.py` - Empirical spawn testing module
- `ground_marker_parser.py` - Parser for RuneLite ground marker data
- `main.py` - Main script to run the optimizer

## Installation

1. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

2. Prepare your walkable tiles data file (a text file with a 2D grid of 0s and 1s)

## Basic Usage

To find the optimal position:

```
python main.py --walkable walkable_tiles.txt
```

Options:

- `--walkable`: Path to walkable tiles data file (default: "walkable_tiles.txt")
- `--radii`: Range of radii to test (e.g., "5-20" or "5,10,15,20") (default: "5-20")
- `--tile-distance`: Use tile distance instead of Euclidean distance (default: True)
- `--parallel`: Use parallel processing for faster computation (default: False)

## Empirical Spawn Testing

You can use your own empirical spawn data to validate or improve the model:

### Converting RuneLite Ground Markers to Spawn Data

If you have ground markers from RuneLite:

1. Export the ground markers JSON data to a file
2. Run the ground marker parser:
   ```
   python ground_marker_parser.py markers.json --output spawns.csv
   ```

This will create a CSV file with the spawn data in the format expected by the testing module.

### Using the Spawn Testing Module

Run the optimizer with spawn testing enabled:

```
python main.py --walkable walkable_tiles.txt --test-spawns --spawn-data spawns.csv --test-pos-x 20 --test-pos-y 31
```

This will:

1. First analyze your empirical spawn data
2. Determine the maximum observed interaction distance
3. Optionally use this empirical radius for optimization
4. Generate visualizations comparing your test data with the computational model

The spawn testing module creates:

- `observed_spawns_visualization.png` - A visualization of your test data
- `test_vs_optimal_comparison.png` - A comparison of your test position vs. the computed optimal position
- `spawn_testing_report.txt` - A detailed report of your spawn observations

## Game Coordinates

To convert map coordinates to in-game coordinates, edit the `convert_to_game_coordinates` function in `spatial_analyser.py`:

```python
def convert_to_game_coordinates(x, y):
    game_x = x + YOUR_X_OFFSET
    game_y = y + YOUR_Y_OFFSET
    return game_x, game_y
```

## Output

The script generates:

- A directory for each radius/tile distance with visualizations
- Summary text file of optimal positions for each radius
- Comparison charts for coverage percentage and visible spawns
- Combined visualization showing all optimal positions
- (If testing enabled) Empirical testing visualization and report

## Example Workflow for OSRS Seaweed Spore Collection

1. Collect data in-game by marking spawn locations with ground markers
2. Export the ground markers data to a JSON file
3. Convert the ground markers to spawn data: `python ground_marker_parser.py markers.json`
4. Run the optimizer with your empirical data: `python main.py --walkable walkable_tiles.txt --test-spawns --spawn-data spawns.csv --test-pos-x YOUR_X --test-pos-y YOUR_Y`
5. Review the results to find the optimal AFK position
