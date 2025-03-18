##main.py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
import sys

# Import the SpawnOptimizer class from spatial_analyser.py
from spatial_analyser import SpawnOptimizer, convert_to_game_coordinates

# Add necessary methods to the SpawnOptimizer class
# These imports add the methods to the class (monkey patching)
from euclidean_distance import (
    find_optimal_position,
    _count_visible_spawns,
    _count_visible_spawns_optimized,
    _find_reachable_tiles,
)

from tile_distance import (
    _find_reachable_tiles_tile_distance,
    _count_visible_spawns_tile_distance,
    find_optimal_position_tile_distance,
)

from visualization import visualize_coverage, visualize_map_with_optimal

from highlighted_map import create_highlighted_tile_map, _create_magnified_view

from other_functions import (
    calculate_best_positions_by_sector,
    create_combined_optimal_locations_graph,
    create_zoomed_optimal_locations_graph,
)

# Global reference to optimizer for use by the testing module
optimizer = None


def run_optimizer_for_radius(optimizer, radius, use_tile_distance=True):
    """
    Run the optimizer for a specific radius and generate visualizations
    with improved titles and game coordinates

    :param optimizer: SpawnOptimizer instance
    :param radius: View radius/tile distance to analyze
    :param use_tile_distance: If True, use tile distance instead of Euclidean distance
    :return: Tuple of (optimal_x, optimal_y, game_x, game_y, coverage, coverage_percentage)
    """
    print(f"\n{'='*60}")
    distance_type = "TILE DISTANCE" if use_tile_distance else "VIEW RADIUS"
    print(f"ANALYZING FOR {distance_type}: {radius}")
    print(f"{'='*60}")

    # Create output directory for this radius
    output_dir = f"{distance_type.lower().replace(' ', '_')}_{radius}_results"
    os.makedirs(output_dir, exist_ok=True)

    # Find optimal position for this radius
    print(f"Finding optimal position with {distance_type.lower()} {radius}...")

    if use_tile_distance:
        optimal_x, optimal_y, coverage = optimizer.find_optimal_position_tile_distance(
            max_tiles=radius,
            use_parallel=False,  # Disable parallel processing to avoid pickle error
        )
    else:
        optimal_x, optimal_y, coverage = optimizer.find_optimal_position(
            view_radius=radius,
            use_parallel=False,  # Disable parallel processing to avoid pickle error
        )

    # Convert to game coordinates
    game_x, game_y = convert_to_game_coordinates(optimal_x, optimal_y)

    # Calculate coverage percentage
    total_spawns = np.sum(optimizer.map)
    coverage_percentage = coverage / total_spawns * 100

    print(f"\nResults for {distance_type.lower()} {radius}:")
    print(f"Optimal Position: ({optimal_x}, {optimal_y})")
    print(f"Game Coordinates: ({game_x}, {game_y})")
    print(f"Visible spawns: {coverage}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")

    # Create visualizations with improved titles
    print(f"\nGenerating visualizations for {distance_type.lower()} {radius}...")

    # Visualize map with optimal position
    plt.figure(figsize=(15, 12))
    visualization = optimizer.visualize_map_with_optimal(
        (optimal_x, optimal_y),
        view_radius=radius if not use_tile_distance else None,
        max_tiles=radius if use_tile_distance else None,
        use_tile_distance=use_tile_distance,
    )
    plt.title(
        f"Spawn Map - {distance_type} {radius}\n"
        f"Optimal Position: ({optimal_x}, {optimal_y}) - Game: ({game_x}, {game_y})"
    )
    plt.savefig(f"{output_dir}/map_visualization.png")
    plt.close()

    # Visualize coverage heatmap
    plt.figure(figsize=(15, 12))
    optimizer.visualize_coverage(
        (optimal_x, optimal_y),
        view_radius=radius if not use_tile_distance else None,
        max_tiles=radius if use_tile_distance else None,
        use_tile_distance=use_tile_distance,
    )
    plt.title(
        f"Coverage Heatmap - {distance_type} {radius}\n"
        f"Position: ({optimal_x}, {optimal_y}) - Game: ({game_x}, {game_y})\n"
        f"Visible Spawns: {coverage} ({coverage_percentage:.1f}%)"
    )
    plt.savefig(f"{output_dir}/coverage_heatmap.png")
    plt.close()

    # Create highlighted tile map
    output_path = f"{output_dir}/highlighted_tiles.png"
    highlighted_map = optimizer.create_highlighted_tile_map(
        (optimal_x, optimal_y),
        view_radius=radius if not use_tile_distance else None,
        max_tiles=radius if use_tile_distance else None,
        game_coordinates=(game_x, game_y),
        use_tile_distance=use_tile_distance,
    )

    # Move generated files to radius directory
    if os.path.exists("highlighted_spawn_tiles.png"):
        os.rename("highlighted_spawn_tiles.png", output_path)

    if os.path.exists("magnified_optimal_position.png"):
        os.rename(
            "magnified_optimal_position.png",
            f"{output_dir}/magnified_view.png",
        )

    if os.path.exists("optimal_position_map.png"):
        os.rename(
            "optimal_position_map.png",
            f"{output_dir}/optimal_position_map.png",
        )

    if os.path.exists("optimal_position_heatmap.png"):
        os.rename(
            "optimal_position_heatmap.png",
            f"{output_dir}/optimal_position_heatmap.png",
        )

    print(f"Visualizations saved to {output_dir}/")

    return (optimal_x, optimal_y, game_x, game_y, coverage, coverage_percentage)


def main():
    """Run the optimizer for multiple radii/tile distances and summarize results"""
    global optimizer  # Use global reference for testing module

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spawn Location Optimizer")
    parser.add_argument(
        "--walkable",
        type=str,
        default="walkable_tiles.txt",
        help="Path to walkable tiles data file",
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="5-20",
        help='Range of radii to test (e.g., "5-20" or "5,10,15,20")',
    )
    parser.add_argument(
        "--tile-distance",
        action="store_true",
        default=True,
        help="Use tile distance instead of Euclidean distance",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="Use parallel processing"
    )

    # Add new arguments for spawn testing
    parser.add_argument(
        "--test-spawns",
        action="store_true",
        default=False,
        help="Enable spawn testing analysis",
    )
    parser.add_argument(
        "--spawn-data",
        type=str,
        default=None,
        help="Path to CSV file with spawn observations",
    )
    parser.add_argument(
        "--test-pos-x", type=int, default=None, help="X coordinate of the test position"
    )
    parser.add_argument(
        "--test-pos-y", type=int, default=None, help="Y coordinate of the test position"
    )

    args = parser.parse_args()

    # Load walkable tiles data
    print(f"Loading walkable tiles from {args.walkable}...")
    walkable_data = np.loadtxt(args.walkable)
    print(f"Loaded walkable tiles with shape {walkable_data.shape}")
    print(f"Total walkable tiles: {np.sum(walkable_data)}")

    # Create optimizer
    optimizer = SpawnOptimizer(walkable_data, walkable_data)

    # Check if we should run spawn testing first
    empirical_radius = None
    if args.test_spawns:
        try:
            from spawn_testing import add_spawn_testing_to_main

            # Set up test position if provided
            test_position = None
            if args.test_pos_x is not None and args.test_pos_y is not None:
                test_position = (args.test_pos_x, args.test_pos_y)

            # Run the testing module first to get empirical radius
            print("\nRunning spawn testing analysis first...")
            empirical_radius = add_spawn_testing_to_main(
                input_file=args.spawn_data,
                test_position=test_position,
                optimizer=optimizer,  # Pass the optimizer explicitly
            )

            if empirical_radius:
                print(
                    f"\nEmpirical testing found maximum interaction radius of {empirical_radius}"
                )

                # Ask user if they want to use the empirical radius
                use_empirical = (
                    input(
                        f"Use empirical radius {empirical_radius} for optimization? (y/n): "
                    )
                    .strip()
                    .lower()
                )
                if use_empirical.startswith("y"):
                    print(f"Using empirical radius {empirical_radius} for optimization")
                    # Override the radii argument with the empirical radius
                    args.radii = str(empirical_radius)
                else:
                    print("Continuing with specified radii")
        except ImportError:
            print("Warning: Spawn testing module not found. Skipping spawn testing.")

    distance_type = "Tile Distance" if args.tile_distance else "View Radius"
    print(f"Using {distance_type} for analysis")

    # Parse radii/distances to test
    if "-" in args.radii:
        start, end = map(int, args.radii.split("-"))
        distances = range(start, end + 1)
    else:
        distances = [int(r) for r in args.radii.split(",")]

    print(f"Analyzing for {distance_type}s: {list(distances)}")

    # Store results for comparison
    results = []

    # Run analysis for each radius/distance
    for distance in distances:
        result = run_optimizer_for_radius(optimizer, distance, args.tile_distance)
        results.append((distance,) + result)

    # Create summary report
    print("\n\nSUMMARY OF RESULTS")
    print("=" * 100)
    print(
        f"{distance_type:<12}{'Map Position':<20}{'Game Position':<22}{'Visible Spawns':<20}{'Coverage %':<15}"
    )
    print("-" * 100)

    for distance, opt_x, opt_y, game_x, game_y, coverage, percentage in results:
        print(
            f"{distance:<12}({opt_x:<3}, {opt_y:<3}){'':>10}({game_x:<5}, {game_y:<5}){'':>8}{coverage:<20}{percentage:<15.2f}"
        )

    # Save summary to file
    with open(
        f"{distance_type.lower().replace(' ', '_')}_analysis_summary.txt", "w"
    ) as f:
        f.write(f"SUMMARY OF {distance_type.upper()} ANALYSIS\n")
        f.write("=" * 100 + "\n")
        f.write(
            f"{distance_type:<12}{'Map Position':<20}{'Game Position':<22}{'Visible Spawns':<20}{'Coverage %':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for distance, opt_x, opt_y, game_x, game_y, coverage, percentage in results:
            f.write(
                f"{distance:<12}({opt_x:<3}, {opt_y:<3}){'':>10}({game_x:<5}, {game_y:<5}){'':>8}{coverage:<20}{percentage:<15.2f}\n"
            )

    # Create comparison charts with improved labels
    plt.figure(figsize=(12, 8))
    distances_values = [r[0] for r in results]
    coverage_values = [r[6] for r in results]

    plt.plot(distances_values, coverage_values, "o-", linewidth=2, markersize=8)
    plt.xlabel(distance_type)
    plt.ylabel("Coverage Percentage (%)")
    plt.title(f"Spawn Coverage by {distance_type}")
    plt.grid(True)
    plt.xticks(distances_values)
    plt.ylim(0, 100)

    plt.savefig(f"{distance_type.lower().replace(' ', '_')}_comparison_chart.png")
    plt.close()

    # Create a visible spawns chart
    plt.figure(figsize=(12, 8))
    spawn_values = [r[5] for r in results]

    plt.plot(
        distances_values, spawn_values, "o-", linewidth=2, markersize=8, color="green"
    )
    plt.xlabel(distance_type)
    plt.ylabel("Number of Visible Spawns")
    plt.title(f"Visible Spawns by {distance_type}")
    plt.grid(True)
    plt.xticks(distances_values)

    plt.savefig(f"visible_spawns_{distance_type.lower().replace(' ', '_')}_chart.png")
    plt.close()

    # Create combined graph with all optimal positions
    create_combined_optimal_locations_graph(results, optimizer)

    # If we ran spawn testing and got results, run the comparison with the optimal position
    if args.test_spawns and empirical_radius and len(results) > 0:
        try:
            from spawn_testing import add_spawn_testing_to_main

            # Get the optimal position from the results
            # Use the first result if multiple radii were analyzed
            optimal_radius = results[0][0]
            optimal_x = results[0][1]
            optimal_y = results[0][2]

            # Run comparison between empirical and computational results
            test_pos = (
                (args.test_pos_x, args.test_pos_y)
                if args.test_pos_x and args.test_pos_y
                else None
            )
            add_spawn_testing_to_main(
                input_file=args.spawn_data,
                test_position=test_pos,
                optimal_position=(optimal_x, optimal_y),
                optimal_radius=optimal_radius,
            )
        except ImportError:
            print("Warning: Spawn testing module not found. Skipping comparison.")

    print("\nProcess complete! Results saved to individual directories.")
    print(
        f"Summary saved to {distance_type.lower().replace(' ', '_')}_analysis_summary.txt"
    )
    print(
        f"Comparison charts saved to {distance_type.lower().replace(' ', '_')}_comparison_chart.png and visible_spawns_{distance_type.lower().replace(' ', '_')}_chart.png"
    )
    print(
        f"Combined optimal positions graph saved to combined_optimal_positions.png and zoomed_optimal_positions.png"
    )


if __name__ == "__main__":
    main()
