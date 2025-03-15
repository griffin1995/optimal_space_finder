import numpy as np
from spatial_analyser import SpawnOptimizer

# Load walkable tiles data
print("Loading walkable tiles...")
walkable_data = np.loadtxt('walkable_tiles.txt')
print(f"Loaded walkable tiles with shape {walkable_data.shape}")
print(f"Total walkable tiles: {np.sum(walkable_data)}")

# Create optimizer
optimizer = SpawnOptimizer(walkable_data, walkable_data)

# Find optimal position
print("\nFinding optimal position...")
optimal_x, optimal_y, coverage = optimizer.find_optimal_position(view_radius=10)

print(f"\nResults:")
print(f"Optimal Position: ({optimal_x}, {optimal_y})")
print(f"Visible spawns: {coverage}")
print(f"Coverage percentage: {coverage/np.sum(walkable_data)*100:.2f}%")

# Create visualizations
print("\nGenerating visualizations...")
optimizer.visualize_map_with_optimal((optimal_x, optimal_y), view_radius=10)
optimizer.visualize_coverage((optimal_x, optimal_y), view_radius=10)
optimizer.create_highlighted_tile_map((optimal_x, optimal_y), view_radius=10)

print("\nProcess complete! Check the output images.")