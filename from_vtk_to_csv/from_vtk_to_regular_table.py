from pyproj import Proj
import vtk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def read_vtk(vtk_file):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    data = reader.GetOutput()
    return data

def get_vtk_bounds(data):
    bounds = data.GetBounds()
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    zmin, zmax = bounds[4], bounds[5]
    return xmin, xmax, ymin, ymax, zmin, zmax

def get_data_values_from_vtk_plane(xy_bounds, x_points, y_points, z_value, data):
    xmin, xmax, ymin, ymax = xy_bounds[0], xy_bounds[1], xy_bounds[2], xy_bounds[3]
    points = vtk.vtkPoints()
    x_vals = np.linspace(xmin, xmax, x_points)
    y_vals = np.linspace(ymin, ymax, y_points) 
    
    for x in x_vals:
        for y in y_vals:
            points.InsertNextPoint(x, y, z_value)
            
    probe_points = vtk.vtkPolyData()
    probe_points.SetPoints(points)
    
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(probe_points)
    probe_filter.SetSourceData(data)
    probe_filter.Update()

    interpolated_data = probe_filter.GetOutput()
    points = interpolated_data.GetPoints()
    values = interpolated_data.GetPointData().GetScalars() 
    
    num_points = points.GetNumberOfPoints()
    # coords = np.array([points.GetPoint(i) for i in range(num_points)])
    data_values = np.array([values.GetValue(i) for i in range(num_points)])

    return data_values
    
def utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon, lat = utm_proj(x, y, inverse=True)
    return lon, lat

def generate_steps(zbeg, zend, dz):
    step_count = (zend - zbeg) / dz
    z_arr = np.arange(zbeg, zend + dz if step_count.is_integer() else zend, dz)
    return z_arr

def output_model_csv(abs_list, pert_list, lon_arr, lat_arr, dep_arr, output_dir):
    global dlon, dlat, dz
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/model.xyz'
    lon_min, lon_max = np.min(lon_arr), np.max(lon_arr)
    lat_min, lat_max = np.min(lat_arr), np.max(lat_arr)
    nx, ny, nz = lon_arr.size, lat_arr.size, dep_arr.size
    
    # ----------------------------------------------------
    # transfering the depth array
    # ----------------------------------------------------
    ddep = dz * -1 / 1E+03
    dep_arr = dep_arr * -1 / 1E+03
    dep_min, dep_max = np.min(dep_arr), np.max(dep_arr)
    # ----------------------------------------------------
    dep_grid, lon_grid, lat_grid = np.meshgrid(dep_arr, lon_arr, lat_arr, indexing='ij')

    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    dep_flat = dep_grid.flatten()
    vp_flat = abs_list[0].flatten()
    vs_flat = abs_list[1].flatten()
    rho_flat = abs_list[2].flatten()
    dvp_flat = pert_list[0].flatten()
    dvs_flat = pert_list[1].flatten()
    drho_flat = pert_list[2].flatten()
    
    vp_min, vp_max = np.nanmin(vp_flat), np.nanmax(vp_flat)
    vs_min, vs_max = np.nanmin(vs_flat), np.nanmax(vs_flat)
    rho_min, rho_max = np.nanmin(rho_flat), np.nanmax(rho_flat)
    dvp_min, dvp_max = np.nanmin(dvp_flat), np.nanmax(dvp_flat)
    dvs_min, dvs_max = np.nanmin(dvs_flat), np.nanmax(dvs_flat)
    drho_min, drho_max = np.nanmin(drho_flat), np.nanmax(drho_flat)

    output_data = np.column_stack((lon_flat, lat_flat, dep_flat, vp_flat, vs_flat, rho_flat, dvp_flat, dvs_flat, drho_flat))
    header_info =  f'{lon_min:.3f} {lat_min:.3f} {dep_min:.3f} {lon_max:.3f} {lat_max:.3f} {dep_max:.3f}\n'
    header_info += f' {dlon:.3f} {dlat:.3f} {ddep:.3f}\n'
    header_info += f' {nx:4d} {ny:4d} {nz:4d}\n'
    header_info += f' {vp_min:.3f} {vp_max:.3f} {vs_min:.3f} {vs_max:.3f} {rho_min:.3f} {rho_max:.3f}\n'
    header_info += f' {dvp_min:.1f} {dvp_max:.1f} {dvs_min:.1f} {dvs_max:.1f} {drho_min:.1f} {drho_max:.1f}'

    np.savetxt(output_file, output_data, fmt='%.3f', header=header_info, comments='')
# ------------------------------
# PARAMETERS
# ------------------------------
target_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_from_ADJOINT_TOMO'
out_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_final'
utm_zone = 50
dlon, dlat = 0.08, 0.08
dz = -5000. # meter (here usually use negative value)
zbeg = 0
scalar_list = ['vp', 'vs', 'rho']
# ------------------------------

if __name__ == "__main__":
    
    model_n = sys.argv[1]
    model_n = int(model_n)
    model_num = f'm{model_n:03d}'
    output_dir = f'{out_dir}/{model_num}'
    vtk_dir = f'{target_dir}/{model_num}/MOD/{scalar_list[0]}.vtk'

    data = read_vtk(vtk_dir)
    xmin, xmax, ymin, ymax, zmin, zmax = get_vtk_bounds(data)
    lon_bound, lat_bound = utm_to_lonlat([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], utm_zone, True)
    lon_min, lon_max = round(min(lon_bound), 3), round(max(lon_bound), 3)
    lat_min, lat_max = round(min(lat_bound), 3), round(max(lat_bound), 3)
    
    lon_arr = np.arange(lon_min, lon_max, dlon)
    lat_arr = np.arange(lat_min, lat_max, dlat)
    lon_points = len(lon_arr)
    lat_points = len(lat_arr)
    
    zend = zmin # negative value stands for deeper part
    dep_arr = generate_steps(zbeg, zend, dz)
    
    # Create the regular grids for coordinates
    # lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr, indexing='ij')
    # lon_grid_flatten, lat_grid_flatten = lon_grid.flatten(), lat_grid.flatten()

    xy_bounds = [xmin, xmax, ymin, ymax]
    
    abs_scalar_list, pert_scalar_list = [], []
    for scalar in scalar_list:
        vtk_dir = f'{target_dir}/{model_num}/MOD/{scalar}.vtk'
        data = read_vtk(vtk_dir)
        
        abs_tmp_list, pert_tmp_list = [], []
        for dep in dep_arr:
            data_values = get_data_values_from_vtk_plane(xy_bounds, lon_points, lat_points, dep, data)
            data_values[data_values == 0.] = np.nan
            data_values = data_values / 1E+03
            mean_in_this_dep = np.nanmean(data_values)
            data_values_pert = (data_values - mean_in_this_dep) / mean_in_this_dep * 1E+02
            abs_tmp_list.append(data_values)
            pert_tmp_list.append(data_values_pert)
            
        abs_values_flatten = np.hstack(abs_tmp_list)
        pert_values_flatten = np.hstack(pert_tmp_list)
        abs_scalar_list.append(abs_values_flatten)
        pert_scalar_list.append(pert_values_flatten)
    
    output_model_csv(abs_scalar_list, pert_scalar_list, lon_arr, lat_arr, dep_arr, output_dir)
    
    
    
    
    


