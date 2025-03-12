# OptimalSpaceFinder

## Overview

OptimalSpaceFinder is a flexible Python utility for analysing 2D spatial data to identify the most efficient standing or positioning location within a defined area. Whether you're optimising game strategies, analysing spatial distributions, or conducting research, this tool provides powerful insights into spatial positioning.

## Key Features

- 🗺️ Multi-Format Map Imports
  - CSV
  - Image (PNG/JPG)
  - JSON
  - Text file

- 📍 Strategic Position Analysis
  - Calculates optimal standing/positioning location
  - Maximises coverage or accessibility
  - Customisable evaluation metrics

- 📊 Advanced Visualisation
  - Heatmap rendering of spatial data
  - Detailed location analysis
  - Multiple visualisation options

## Potential Use Cases

- 🎮 Game Strategy Optimisation
  - Resource collection
  - Area coverage
  - Spawn point analysis

- 🏭 Facility Layout Planning
  - Warehouse positioning
  - Network coverage
  - Sensor placement

- 🌍 Geospatial Analysis
  - Resource distribution
  - Coverage mapping
  - Accessibility studies

## Prerequisites

### System Requirements
- Python 3.8+
- pip package manager

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn pillow
```

## Installation

```bash
# Clone the repository
git clone https://github.com/griffin1995/optimal_space_finder.git

# Change directory
cd optimal_space_finder

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from spatial_analyser import SpatialAnalyser
from map_import_methods import MapImporter

# Import your spatial map
map_array = MapImporter.from_image('your_spatial_map.png')

# Create analyser
analyser = SpatialAnalyser(map_array)

# Find optimal position
optimal_x, optimal_y = analyser.find_optimal_position()
print(f"Optimal Position: (x={optimal_x}, y={optimal_y})")

# Visualise spatial distribution
analyser.visualise_spatial_distribution()
```

## Supported Input Formats

### CSV
```
0,0,0,0
0,1,1,0
0,1,0,0
```

### Image
- Black and white or greyscale image
- Pixel intensity represents spatial significance

### JSON
```json
{
  "spatial_data": [
    [0,0,0,0],
    [0,1,1,0],
    [0,1,0,0]
  ]
}
```

### Text
```
....
.XX.
.X..
```

## Customisation

- Adjust spatial significance threshold
- Customise analysis metrics
- Modify visualisation styles
- Extend for specific use cases

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/advanced_analysis`)
3. Commit your changes (`git commit -m 'Add advanced positioning algorithm'`)
4. Push to the branch (`git push origin feature/advanced_analysis`)
5. Open a Pull Request

## Extensibility

The modular design allows easy extension for:
- Custom positioning algorithms
- Additional import methods
- Specialised visualisation techniques

## Licence

Distributed under the MIT Licence. See `LICENCE` for more information.

## Contact

Project Link: [https://github.com/griffin1995/optimal_space_finder](https://github.com/griffin1995/optimal_space_finder)

## Disclaimer

This is a generic tool designed to be adaptable to various spatial analysis scenarios.
