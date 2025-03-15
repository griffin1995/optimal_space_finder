import numpy as np
import os
import matplotlib.pyplot as plt
from spatial_analyser import SpawnOptimizer

def run_optimizer_for_radius(optimizer, radius):
    """
    Run the optimizer for a specific radius and generate visualizations
    
    :param optimizer: SpawnOptimizer instance
    :param radius: View radius to analyze
    :return: Tuple of (optimal_x, optimal_y, coverage, coverage_percentage)
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING FOR RADIUS: {radius}")
    print(f"{'='*60}")
    
    # Create output directory for this radius
    output_dir = f"radius_{radius}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find optimal position for this radius - DISABLING PARALLEL PROCESSING
    print(f"Finding optimal position with radius {radius}...")
    optimal_x, optimal_y, coverage = optimizer.find_optimal_position(
        view_radius=radius, 
        use_parallel=False  # Disable parallel processing to avoid pickle error
    )
    
    # Calculate coverage percentage
    total_spawns = np.sum(optimizer.map)
    coverage_percentage = coverage / total_spawns * 100
    
    print(f"\nResults for radius {radius}:")
    print(f"Optimal Position: ({optimal_x}, {optimal_y})")
    print(f"Visible spawns: {coverage}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")
    
    # Create visualizations with radius in the title
    print(f"\nGenerating visualizations for radius {radius}...")
    
    # Visualize map with optimal position
    plt.figure(figsize=(15, 12))
    visualization = optimizer.visualize_map_with_optimal((optimal_x, optimal_y), view_radius=radius)
    plt.title(f"Spawn Map - Radius {radius} - Optimal Position: ({optimal_x}, {optimal_y})")
    plt.savefig(f"{output_dir}/map_visualization_r{radius}.png")
    plt.close()
    
    # Visualize coverage heatmap
    plt.figure(figsize=(15, 12))
    optimizer.visualize_coverage((optimal_x, optimal_y), view_radius=radius)
    plt.title(f"Coverage Heatmap - Radius {radius} - Visible Spawns: {coverage} ({coverage_percentage:.1f}%)")
    plt.savefig(f"{output_dir}/coverage_heatmap_r{radius}.png")
    plt.close()
    
    # Create highlighted tile map
    output_path = f"{output_dir}/highlighted_tiles_r{radius}.png"
    highlighted_map = optimizer.create_highlighted_tile_map(
        (optimal_x, optimal_y), 
        view_radius=radius
    )
    
    # Move generated files to radius directory
    if os.path.exists("highlighted_spawn_tiles.png"):
        os.rename("highlighted_spawn_tiles.png", output_path)
        
    if os.path.exists("magnified_optimal_position.png"):
        os.rename("magnified_optimal_position.png", f"{output_dir}/magnified_view_r{radius}.png")
    
    if os.path.exists("optimal_position_map.png"):
        os.rename("optimal_position_map.png", f"{output_dir}/optimal_position_map_r{radius}.png")
        
    if os.path.exists("optimal_position_heatmap.png"):
        os.rename("optimal_position_heatmap.png", f"{output_dir}/optimal_position_heatmap_r{radius}.png")
    
    print(f"Visualizations saved to {output_dir}/")
    
    return (optimal_x, optimal_y, coverage, coverage_percentage)

def main():
    """Run the optimizer for multiple radii and summarize results"""
    # Load walkable tiles data
    print("Loading walkable tiles...")
    walkable_data = np.loadtxt('walkable_tiles.txt')
    print(f"Loaded walkable tiles with shape {walkable_data.shape}")
    print(f"Total walkable tiles: {np.sum(walkable_data)}")
    
    # Create optimizer
    optimizer = SpawnOptimizer(walkable_data, walkable_data)
    
    # Define radii to test
    radii = range(5, 21)  # 5 to 20 inclusive
    
    # Store results for comparison
    results = []
    
    # Run analysis for each radius
    for radius in radii:
        result = run_optimizer_for_radius(optimizer, radius)
        results.append((radius,) + result)
    
    # Create summary report
    print("\n\nSUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Radius':<10}{'Position':<20}{'Visible Spawns':<20}{'Coverage %':<15}")
    print("-" * 80)
    
    for radius, opt_x, opt_y, coverage, percentage in results:
        print(f"{radius:<10}({opt_x:<3}, {opt_y:<3}){'':>10}{coverage:<20}{percentage:<15.2f}")
    
    # Save summary to file
    with open("radius_analysis_summary.txt", "w") as f:
        f.write("SUMMARY OF RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Radius':<10}{'Position':<20}{'Visible Spawns':<20}{'Coverage %':<15}\n")
        f.write("-"*80 + "\n")
        
        for radius, opt_x, opt_y, coverage, percentage in results:
            f.write(f"{radius:<10}({opt_x:<3}, {opt_y:<3}){'':>10}{coverage:<20}{percentage:<15.2f}\n")
    
    # Create comparison chart
    plt.figure(figsize=(12, 8))
    radii_values = [r[0] for r in results]
    coverage_values = [r[3] for r in results]
    
    plt.plot(radii_values, coverage_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('View Radius')
    plt.ylabel('Coverage Percentage (%)')
    plt.title('Spawn Coverage by View Radius')
    plt.grid(True)
    plt.xticks(radii_values)
    plt.ylim(0, 100)
    
    plt.savefig("radius_comparison_chart.png")
    
    # Create a coverage chart
    plt.figure(figsize=(12, 8))
    spawn_values = [r[2] for r in results]
    
    plt.plot(radii_values, spawn_values, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('View Radius')
    plt.ylabel('Number of Visible Spawns')
    plt.title('Visible Spawns by View Radius')
    plt.grid(True)
    plt.xticks(radii_values)
    
    plt.savefig("visible_spawns_chart.png")
    
    print("\nProcess complete! Results saved to individual radius directories.")
    print("Summary saved to radius_analysis_summary.txt")
    print("Comparison charts saved to radius_comparison_chart.png and visible_spawns_chart.png")

if __name__ == "__main__":
    main()