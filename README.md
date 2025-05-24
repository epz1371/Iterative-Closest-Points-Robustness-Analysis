# Iterative Closest Points (ICP) Implementation

This project implements the Iterative Closest Points (ICP) algorithm for 3D point cloud registration. It provides a robust solution for aligning two point clouds by iteratively finding the best transformation between them.

## Features

- Multiple setup options for point cloud registration:
  - Custom single point registration
  - Custom range-based registration
  - 360-degree rotation analysis
- Visualization tools:
  - 3D point cloud visualization
  - Sphere projection of rotation results
  - Histogram analysis of matching accuracy
- Configurable parameters for fine-tuning the registration process
- Support for various point cloud file formats (PLY)

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)
- Open3D for point cloud processing
- PyVista for 3D visualization
- NumPy for numerical computations
- SciPy for spatial operations
- Matplotlib for plotting

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your point cloud files in the data directory
2. Configure the parameters in the script (if needed)
3. Run the script:
```bash
python icp_implementation.py
```

4. Choose one of the three setup options:
   - Custom-One-Point-Setup: Test a specific rotation
   - Custom-View-All-Points-In-Range-Setup: Test a range of rotations
   - 360°-Setup: Test full rotation around an axis

## Configuration

The script includes several configurable parameters:

- `include_every_nth_point_source`: Sampling rate for source point cloud
- `include_every_nth_point_destination`: Sampling rate for destination point cloud
- `option_radius_closest_points_correspondences`: Search radius for point correspondences
- `option_distance_threshold_for_matching_points`: Threshold for point matching
- `option_ICP_iterations`: Maximum number of ICP iterations

## Output

The script generates:
1. Visualization of point clouds before and after registration
2. Sphere projection showing successful and failed rotations
3. Histogram of matching accuracy
4. Detailed statistics for each registration attempt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

©
If you use this project or parts of it, you must cite the author and this GitHub repository as the source in your work.

## Acknowledgments

- Based on the work of Ehsan Pazooki
- Uses Open3D for point cloud processing
- Implements PyVista for visualization 
