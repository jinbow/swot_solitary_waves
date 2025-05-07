"""
Process SWOT Level-2 LR data to create 65x65 pixel segments for machine learning in parallel.

This script extracts segments from SWOT Level-2 Low-Resolution (LR) data, processes
them into 65x65 pixel segments, and saves them as PNGs (zipped) or NetCDF files with uint8
scaling in a single (n_seg, y, x) array. It uses MPI for parallel processing across multiple
NetCDF files organized in cycle folders. Originally developed for one cycle on Eddy@JPL
in November 2023, it was adapted to process all cycles and passes up to April 10, 2025,
on Grace@TAMU.

Added mpi4py for parallel processing, NetCDF output with uint8 scaling, PNG zipping, and
strict 65x65 pixel PNG output without axes. Fixed font rendering and NaN issues.

Author: Jinbo Wang <jinbowang@tamu.edu>
Created: November 2023
Last Updated: April 2025 with assistance from Grok
"""


import os
import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from mpi4py import MPI
import zipfile
from scipy.signal import detrend
from scipy.interpolate import griddata 

def fit_plane_and_get_anomaly(x, y, z):
    # Number of points

    ny,nx=x.shape

    x=x.flatten()
    y=y.flatten()
    z=z.flatten()
    
    m=np.isfinite(z)
    
    zz=z[m]

    #remove extreme values
    vmin, vmax = np.percentile(zz, [0.5, 99.5])
    
    z[m]=np.where( (zz<vmin)|(zz>vmax), np.nan, zz)

    m=np.isfinite(z)
    zz=z[m]
    
    n = zz.size
    
    # Form the design matrix A
    A = np.vstack([x[m]**2, y[m]**2, x[m]*y[m], x[m], y[m], np.ones(n)]).T
    
    # Solve for coefficients [a, b, c] using least-squares
    coeffs, _, _, _ = np.linalg.lstsq(A, zz, rcond=None)
    
    # Extract a, b, c
    a, b, c, d, e, f = coeffs
    
    # Compute predicted z-values: z_pred = ax + by + c
    z_pred = a * x[m]**2 + b * y[m]**2 + c*x[m]*y[m] + d * x[m] + e * y[m] + f
    
    # Compute anomalies: z - z_pred
    anomalies = zz - z_pred
    z[m]=anomalies
    # interpolate 
    #znew = griddata(np.c_[x[m],y[m]], anomalies, (x,y), method='cubic',fill_value=0).reshape(ny,nx)
    
    return z.reshape(ny,nx) #np.c_[znew[:,:25],znew[:,35:-1]]

def parse_filename(input_file: str) -> tuple:
    """
    Extract cycle and pass numbers from input filename.

    Args:
        input_file (str): Path to the NetCDF file.

    Returns:
        tuple: (cycle_number, pass_number)
    """
    base_filename = os.path.basename(input_file)[:-3]
    parts = base_filename.split('_')
    return parts[5], parts[6]

def create_segment_png(
    segment_ssha: np.ndarray,
    cycle_number: str,
    pass_number: str,
    y_start: int,
    output_dir: str,
    fig: plt.Figure,
    ax: plt.Axes
) -> str:
    """
    Create a 65x65 pixel PNG from a 2D SSH anomaly segment using provided figure and axes.

    Args:
        segment_ssha (np.ndarray): 2D array of SSH anomaly data (60x60).
        cycle_number (str): Cycle number (e.g., '001').
        pass_number (str): Pass number (e.g., '001').
        y_start (int): Starting y-index for segment naming.
        output_dir (str): Directory to save the PNG file.
        fig (plt.Figure): Matplotlib figure to reuse.
        ax (plt.Axes): Matplotlib axes to reuse.

    Returns:
        str: Path to the created PNG file, or empty string if creation fails.
    """
    try:
        ax.clear()  # Clear previous plot
        vmin, vmax = np.percentile(segment_ssha[np.isfinite(segment_ssha)], [0.5, 99.5])
        ax.imshow(segment_ssha, cmap='binary_r', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        output_filename = f"{cycle_number}_{pass_number}_{y_start:04d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        fig.savefig(
            output_path,
            format='png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0
        )
        return output_path

    except Exception as e:
        print(f"Rank {comm.Get_rank()} - Error creating PNG {output_filename}: {e}")
        return ""

def process_swot_segments(
    input_file: str,
    output_dir: str,
    validate_time: bool = True,
    output_format: str = 'png'
) -> None:
    
    """
    Process a single SWOT Level-2 LR NetCDF file to extract 65x65 pixel segments and save as PNG or NetCDF.

    For 'netcdf', segments are saved in a single (n_seg, y, x) uint8 array with metadata.
    For 'png', PNGs are generated as 65x65 pixel images with no axes, ticks, or labels, then
    zipped into one .zip file, and individual PNGs are deleted.
    For 'both', both outputs are produced.

    Args:
        input_file (str): Path to the input NetCDF file containing SWOT data.
        output_dir (str): Directory where output files (PNG zips or NetCDF) will be saved.
        validate_time (bool, optional): If True, checks for valid time data. Defaults to True.
        output_format (str, optional): Output format: 'png', 'netcdf', or 'both'. Defaults to 'png'.

    Returns:
        None
    """
    
    if output_format not in ['png', 'netcdf', 'both']:
        print(f"Rank {comm.Get_rank()} - Invalid output_format: {output_format}. Skipping {input_file}")
        return

    try:
        with Dataset(input_file, 'r') as dataset:
            ssha = dataset.variables['ssha_karin_2'][:].filled(np.nan)
            longitude = dataset.variables['longitude'][:].filled(np.nan)
            latitude = dataset.variables['latitude'][:].filled(np.nan)
            time = dataset.variables['time'][:].filled(np.nan)
            quality_mask = dataset.variables['ssha_karin_2_qual'][:] == 0
            surface_mask = dataset.variables['ancillary_surface_classification_flag'][:] == 0
    except Exception as e:
        print(f"Rank {comm.Get_rank()} - Error opening {input_file}: {e}")
        return

    valid_mask = quality_mask & surface_mask
    ssha = np.where(valid_mask, ssha, np.nan)

    segment_size = 64
    num_lines = ssha.shape[0]
    num_segments = num_lines // segment_size

    cycle_number, pass_number = parse_filename(input_file)
    base_filename = os.path.basename(input_file)[:-3]
    xx,yy=np.meshgrid(np.arange(segment_size),np.arange(segment_size))
    
    # Initialize figure for PNG output
    fig = ax = None
    if output_format in ['png', 'both']:
        fig = plt.figure(figsize=(0.64, 0.64), dpi=100)
        ax = fig.add_subplot(111)
        
        png_files = []

    # Initialize NetCDF storage
    segments = scale_factors = add_offsets = None
    if output_format in ['netcdf', 'both']:
        segments = []
        scale_factors = []
        add_offsets = []

    for segment_idx in range(num_segments):
        y_start = segment_idx * segment_size
        y_end = y_start + segment_size
        x_start = 3
        x_end = x_start + segment_size

        segment_ssha = ssha[y_start:y_end, x_start:x_end]
        if np.isfinite(segment_ssha).sum() > segment_ssha.size * 0.80:
            segment_time = time[y_start:y_end]
            time_mean = segment_time[segment_size // 2]

            if validate_time and np.isnan(time_mean):
                print(f"Rank {comm.Get_rank()} - TIME ERROR in {input_file}, segment {segment_idx}")
                continue

            # Detrend efficiently
            #msk=np.isfinite(segment_ssha)
            
            #segment_ssha = np.nan_to_num(segment_ssha, nan=np.nanmean(segment_ssha))
            #segment_ssha = detrend(detrend(segment_ssha, axis=0), axis=1)
            segment_ssha=fit_plane_and_get_anomaly(xx,yy,segment_ssha)
            if output_format in ['netcdf', 'both']:
                vmin, vmax = np.percentile(segment_ssha, [2, 98])
                scale_factor = (vmax - vmin) / 255.0
                add_offset = vmin
                scaled_ssha = np.clip((segment_ssha - vmin) / scale_factor, 0, 255)
                scaled_ssha = np.where(np.isnan(segment_ssha), 255, scaled_ssha).astype(np.uint8)

                segments.append(scaled_ssha)
                scale_factors.append(scale_factor)
                add_offsets.append(add_offset)

            if output_format in ['png', 'both']:
                png_path = create_segment_png(segment_ssha, cycle_number, pass_number, y_start, output_dir, fig, ax)
                if png_path:
                    png_files.append(png_path)

    # Save NetCDF
    if output_format in ['netcdf', 'both'] and segments:
        nc_output_file = os.path.join(output_dir, f"{base_filename}_segments.nc")
        try:
            with Dataset(nc_output_file, 'w', format='NETCDF4') as nc:
                nc.createDimension('n_seg', len(segments))
                nc.createDimension('y', segment_size)
                nc.createDimension('x', segment_size)

                seg_var = nc.createVariable('segments', np.uint8, ('n_seg', 'y', 'x'), fill_value=255, zlib=True)
                scale_var = nc.createVariable('scale_factor', np.float32, ('n_seg',))
                offset_var = nc.createVariable('add_offset', np.float32, ('n_seg',))

                seg_var[:] = np.stack(segments, axis=0)
                scale_var[:] = scale_factors
                offset_var[:] = add_offsets

                seg_var.units = 'meters (scaled)'
                seg_var.description = 'Detrended SSH anomaly scaled to uint8'
                seg_var.scale_factor_description = 'Multiply by scale_factor and add add_offset to get meters'
        except Exception as e:
            print(f"Rank {comm.Get_rank()} - Error writing NetCDF {nc_output_file}: {e}")

    # Zip PNGs
    if output_format in ['png', 'both'] and png_files:
        zip_output_file = os.path.join(output_dir, f"{base_filename}_pngs.zip")
        try:
            with zipfile.ZipFile(zip_output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for png_file in png_files:
                    zipf.write(png_file, os.path.basename(png_file))
                    try:
                        os.remove(png_file)
                    except Exception as e:
                        print(f"Rank {comm.Get_rank()} - Error deleting {png_file}: {e}")
        except Exception as e:
            print(f"Rank {comm.Get_rank()} - Error creating zip {zip_output_file}: {e}")

    if fig:
        plt.close(fig)

def collect_nc_files(base_dir: str, max_files: int = 10) -> list:
    """
    Collect up to max_files .nc files from cycle folders.

    Args:
        base_dir (str): Base directory containing cycle folders.
        max_files (int): Maximum number of files to collect.

    Returns:
        list: List of .nc file paths.
    """
    nc_files = []
    for cycle_folder in sorted(glob.glob(os.path.join(base_dir, 'cycle_???'))):
        nc_files.extend(sorted(glob.glob(os.path.join(cycle_folder, '*.nc'))))
    return nc_files[:max_files]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_dir = "/scratch/group/sat.ocean.lab/swot/expert"
    output_dir = "/scratch/group/sat.ocean.lab/swot/expert/output_segments_4_ML"
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        nc_files = collect_nc_files(base_dir)
        print(f"Found {len(nc_files)} .nc files")
    else:
        nc_files = None

    comm.Barrier()
    nc_files = comm.bcast(nc_files, root=0)

    # Dynamic load balancing
    local_files = np.array_split(nc_files, size)[rank] if nc_files else []

    for input_file in local_files:
        print(f"Rank {rank} processing {input_file}")
        process_swot_segments(input_file, output_dir, validate_time=True, output_format='png')

    comm.Barrier()
    if rank == 0:
        print("All files processed.")
import os
import re
import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpi4py import MPI
import zipfile
from scipy.signal import detrend

def process_swot_segments(
    input_file: str,
    output_dir: str,
    validate_time: bool = True,
    output_format: str = 'png'
) -> None:
    """
    Process a single SWOT Level-2 LR NetCDF file to extract 65x65 pixel segments and save as PNG or NetCDF.

    For 'netcdf', segments are saved in a single (n_seg, y, x) uint8 array with metadata.
    For 'png', PNGs are generated as 65x65 pixel images with no axes, ticks, or labels, then
    zipped into one .zip file, and individual PNGs are deleted.
    For 'both', both outputs are produced.

    Args:
        input_file (str): Path to the input NetCDF file containing SWOT data.
        output_dir (str): Directory where output files (PNG zips or NetCDF) will be saved.
        validate_time (bool, optional): If True, checks for valid time data. Defaults to True.
        output_format (str, optional): Output format: 'png', 'netcdf', or 'both'. Defaults to 'png'.

    Returns:
        None
    """
    if output_format not in ['png', 'netcdf', 'both']:
        print(f"Rank {comm.Get_rank()} - Invalid output_format: {output_format}. Skipping {input_file}")
        return

    # Load dataset with netCDF4
    try:
        dataset = Dataset(input_file, 'r')
    except Exception as e:
        print(f"Rank {comm.Get_rank()} - Error opening {input_file}: {e}")
        return

    # Extract data arrays
    ssha = dataset.variables['ssha_karin_2'][:]  # Sea surface height anomaly
    longitude = dataset.variables['longitude'][:]
    latitude = dataset.variables['latitude'][:]
    time = dataset.variables['time'][:]
    quality_mask = dataset.variables['ssha_karin_2_qual'][:] == 0  # Quality flag
    surface_mask = dataset.variables['ancillary_surface_classification_flag'][:] == 0  # Surface classification

    # Close dataset
    dataset.close()

    # Combine masks to filter valid data
    valid_mask = quality_mask & surface_mask
    ssha = np.where(valid_mask, ssha, np.nan)  # Apply mask to SSH data

    # Define segment dimensions
    segment_size_x = 60
    segment_size_y = 60
    num_lines = ssha.shape[0]
    num_segments = num_lines // segment_size_y

    # Lists to store segments and metadata for NetCDF
    if output_format in ['netcdf', 'both']:
        segments = []
        lat_means = []
        lon_means = []
        time_means = []
        scale_factors = []
        add_offsets = []

    # Initialize plot if PNG output is needed
    if output_format in ['png', 'both']:
        # Set figure size for 65x65 pixels at DPI=100
        fig = plt.figure(figsize=(0.65, 0.65), dpi=100)
        ax = fig.add_subplot(111)
        # Remove all axes decorations
        ax.set_axis_off()  # Turn off axis lines and labels
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_xticklabels([])  # Ensure no tick labels
        ax.set_yticklabels([])  # Ensure no tick labels
        ax.set_frame_on(False)  # Remove spines
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No margins
        png_files = []  # Track PNGs for zipping

    # Extract filename components for output naming
    base_filename = os.path.basename(input_file)[:-3]
    filename_parts = base_filename.split('_')
    file_metadata = [filename_parts[i] for i in [5, 6, 9, 10]]  # Cycle, pass, etc.

    # Process each segment
    for segment_idx in range(num_segments):
        # Define segment boundaries
        y_start = segment_idx * segment_size_y
        y_end = y_start + segment_size_y
        x_start = 4  # Offset to avoid edge effects
        x_end = x_start + segment_size_x

        # Check if segment has sufficient valid data
        segment_ssha = ssha[y_start:y_end, x_start:x_end]
        segment_mask = np.isfinite(segment_ssha)
        
        if segment_mask.sum() > (segment_size_x * segment_size_y) * 0.75:  # Stricter threshold
            # Extract segment data
            segment_lon = longitude[y_start:y_end, x_start:x_end]
            segment_lat = latitude[y_start:y_end, x_start:x_end]
            segment_time = time[y_start:y_end]

            # Calculate mean coordinates and time
            lon_mean = np.nanmean(segment_lon)
            lat_mean = np.nanmean(segment_lat)
            time_mean = segment_time[round(len(segment_time) / 2)]

            # Validate time if requested
            if validate_time and str(time_mean).lower() in ['na', 'nat']:
               print(f"Rank {comm.Get_rank()} - TIME ERROR in {input_file}")
               print(f"t0_mean: {time_mean}")
               print(f"time: {time}")
               print(f"round(len(segment_time)/2): {round(len(segment_time)/2)}")
               continue

            # Format strings for filename
            
            lon_str = re.sub(r'[.]', 'p', str(lon_mean))
            lat_str = re.sub(r'[.]', 'p', str(lat_mean))
            time_str = re.sub(r'[-:]', '', str(time_mean)).split('.')[0]

            # Detrend SSH data, handling NaNs
            # segment_ssha = segment_ssha.copy()  # Avoid modifying original
            segment_ssha=np.where(np.isnan(segment_ssha),np.nanmean(segment_ssha),segment_ssha)
            segment_ssha=detrend(detrend(segment_ssha,axis=0),axis=1)

            #mean_x = np.nanmean(segment_ssha, axis=0, keepdims=True)
            #mean_y = np.nanmean(segment_ssha, axis=1, keepdims=True)
            # Replace NaN means with 0 to avoid all-NaN results
            #mean_x = np.where(np.isnan(mean_x), 0, mean_x)
            #mean_y = np.where(np.isnan(mean_y), 0, mean_y)
            #segment_ssha -= mean_x
            #segment_ssha -= mean_y

            # Calculate 2nd and 98th percentiles for scaling
            vmin = np.percentile(segment_ssha, 2)
            vmax = np.percentile(segment_ssha, 98)
            
            # Handle NetCDF output
            if output_format in ['netcdf', 'both']:
                # Scale to uint8 (0-255)
                scale_factor = (vmax - vmin) / 255.0
                add_offset = vmin
                scaled_ssha = np.clip((segment_ssha - vmin) / scale_factor, 0, 255)
                scaled_ssha = np.where(np.isnan(segment_ssha), 255, scaled_ssha).astype(np.uint8)

                # Store segment and metadata
                segments.append(scaled_ssha)
                #lat_means.append(lat_mean)
                #lon_means.append(lon_mean)
                #time_means.append(str(time_mean))
                scale_factors.append(scale_factor)
                add_offsets.append(add_offset)

            # Handle PNG output
            if output_format in ['png', 'both']:
                output_filename = (f"{file_metadata[0]}_{file_metadata[1]}_{y_start:04d}.png")
                output_path = os.path.join(output_dir, output_filename)
                try:
                    ax.imshow(segment_ssha, cmap='binary_r', vmin=vmin, vmax=vmax)
                    # Remove all axes decorations
                    ax.set_axis_off()  # Turn off axis lines and labels
                    ax.set_xticks([])  # Remove x ticks
                    ax.set_yticks([])  # Remove y ticks
                    ax.set_xticklabels([])  # Ensure no tick labels
                    ax.set_yticklabels([])  # Ensure no tick labels
                    ax.set_frame_on(False)  # Remove spines
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No margins
                    plt.axis('off')
        
                    plt.savefig(
                        output_path,
                        format='png',
                        dpi=100,  # Safe DPI for 65x65 pixels
                        bbox_inches='tight',
                        pad_inches=0
                    )
                    
                    ax.clear()
                    png_files.append(output_path)
                except Exception as e:
                    print(f"Rank {comm.Get_rank()} - Error saving PNG {output_filename}: {e}")
                    continue

    # Save NetCDF if needed
    if output_format in ['netcdf', 'both'] and segments:
        nc_output_file = os.path.join(output_dir, f"{base_filename}_segments.nc")
        try:
            with Dataset(nc_output_file, 'w', format='NETCDF4') as nc_dataset:
                # Create dimensions
                nc_dataset.createDimension('n_seg', len(segments))
                nc_dataset.createDimension('y', segment_size_y)
                nc_dataset.createDimension('x', segment_size_x)

                # Create variables
                segments_var = nc_dataset.createVariable(
                    'segments', np.uint8, ('n_seg', 'y', 'x'), fill_value=255, zlib=True)
                lat_var = nc_dataset.createVariable('lat_mean', np.float32, ('n_seg',))
                lon_var = nc_dataset.createVariable('lon_mean', np.float32, ('n_seg',))
                time_var = nc_dataset.createVariable('time_mean', str, ('n_seg',))
                scale_var = nc_dataset.createVariable('scale_factor', np.float32, ('n_seg',))
                offset_var = nc_dataset.createVariable('add_offset', np.float32, ('n_seg',))

                # Assign data
                segments_var[:] = np.stack(segments, axis=0)
                lat_var[:] = lat_means
                lon_var[:] = lon_means
                time_var[:] = time_means
                scale_var[:] = scale_factors
                offset_var[:] = add_offsets

                # Add attributes
                segments_var.units = 'meters (scaled)'
                segments_var.description = 'Detrended SSH anomaly scaled to uint8'
                segments_var.scale_factor_description = 'Multiply by scale_factor and add add_offset to get meters'
                lat_var.units = 'degrees_north'
                lon_var.units = 'degrees_east'
                time_var.description = 'Mean time of segment'
        except Exception as e:
            print(f"Rank {comm.Get_rank()} - Error writing NetCDF {nc_output_file}: {e}")

    # Zip PNGs and delete originals if needed
    if output_format in ['png', 'both'] and png_files:
        zip_output_file = os.path.join(output_dir, f"{base_filename}_pngs.zip")
        try:
            with zipfile.ZipFile(zip_output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for png_file in png_files:
                    zipf.write(png_file, os.path.basename(png_file))
            # Delete individual PNGs after zipping
            for png_file in png_files:
                try:
                    os.remove(png_file)
                except Exception as e:
                    print(f"Rank {comm.Get_rank()} - Error deleting {png_file}: {e}")
        except Exception as e:
            print(f"Rank {comm.Get_rank()} - Error creating zip {zip_output_file}: {e}")

    # Clean up
    if output_format in ['png', 'both']:
        plt.close(fig)


def collect_nc_files(base_dir: str) -> list:
    """
    Collect all .nc files from cycle folders in the base directory.

    Args:
        base_dir (str): Base directory containing cycle folders (e.g., cycle_001).

    Returns:
        list: List of paths to .nc files.
    """
    nc_files = []
    # Find all cycle folders
    
    cycle_folders = sorted(glob.glob(os.path.join(base_dir, 'cycle_???')))
    
    for cycle_folder in cycle_folders:
        # Find all .nc files in the cycle folder
        files = sorted(glob.glob(os.path.join(cycle_folder, '*.nc')))
        nc_files.extend(files)
        
    return nc_files[:1]


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Base directory and output directory
    base_dir = "/scratch/group/sat.ocean.lab/swot/expert"
    output_dir = "/scratch/group/sat.ocean.lab/swot/expert/output_segments_4_ML"
    
    # Create output directory if it doesn't exist (only by rank 0)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Synchronize processes
    comm.Barrier()

    # Collect all .nc files (only by rank 0)
    if rank == 0:
        nc_files = collect_nc_files(base_dir)
        print(f"Found {len(nc_files)} .nc files")
    else:
        nc_files = None

    # Broadcast the file list to all processes
    nc_files = comm.bcast(nc_files, root=0)

    # Distribute files across processes
    files_per_process = len(nc_files) // size
    remainder = len(nc_files) % size
    start_idx = rank * files_per_process + min(rank, remainder)
    if rank < remainder:
        num_files = files_per_process + 1
    else:
        num_files = files_per_process

    # Calculate file indices for this process
    end_idx = start_idx + num_files
    local_files = nc_files[start_idx:end_idx]

    # Process assigned files
    for input_file in local_files:
        if rank == 0:
            print(f"Rank {rank} processing {input_file}")
        # Change output_format to 'netcdf' or 'both' as needed
        process_swot_segments(input_file, output_dir, validate_time=True, output_format='png')

    # Synchronize processes
    comm.Barrier()
    if rank == 0:
        print("All files processed.")