import numpy as np
import vtk
from vtk.util import numpy_support
from pyproj import Proj
from scipy.interpolate import griddata
import sys
import pandas as pd
import os

def extract_vtk_loc_and_scalar(target_vtk_file, target_scalar='vp'):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(target_vtk_file)
    reader.Update()

    data = reader.GetOutput()
    points = data.GetPoints()
    num_points = points.GetNumberOfPoints()

    points_array = np.zeros((num_points, 3)) 
    for i in range(num_points):
        point = points.GetPoint(i)
        points_array[i] = point
        
    value_scalars = data.GetPointData().GetArray(target_scalar)
    value_array = numpy_support.vtk_to_numpy(value_scalars)
    
    return points_array, value_array

def utm_to_lonlat(x_arr, y_arr, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon_arr, lat_arr = utm_proj(x_arr, y_arr, inverse=True)
    
    return lon_arr, lat_arr

def check_and_create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        pass


if __name__ == "__main__":
    args = sys.argv[1]
    model_n = int(args)
    model_num = f'm{model_n:03d}'
    print(f'model:{model_num}')
    scalar_list = ['vp', 'vs', 'rho']
    vtk_dir = f'/home/harry/Work/FWI_result/output_from_ADJOINT_TOMO/{model_num}/MOD'
    output_dir = f'/home/harry/Work/FWI_result/output_final/{model_num}'
    
    loc_arr, tmp = extract_vtk_loc_and_scalar(target_vtk_file = f'{vtk_dir}/vp.vtk', 
                                                target_scalar = 'vp')
    x_utm_arr, y_utm_arr = loc_arr[:,0], loc_arr[:,1]
    lon_arr, lat_arr = utm_to_lonlat(x_arr = x_utm_arr, y_arr = y_utm_arr,
                                    utm_zone = 50, is_north_hemisphere = True)
    dep_arr = loc_arr[:,2] / 1E+03 * -1.
    data_points_arr = np.array(list(zip(lon_arr, lat_arr, dep_arr)))
    scalar_array = data_points_arr.copy()
    
    for scalar in scalar_list:
        tmp, scalar_arr = extract_vtk_loc_and_scalar(target_vtk_file = f'{vtk_dir}/{scalar}.vtk', 
                                                        target_scalar = f'{scalar}')

        scalar_array = np.column_stack((scalar_array, scalar_arr))

    final_df = pd.DataFrame(scalar_array, columns=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    
    check_and_create_dir(output_dir)
    final_df.to_csv(f'{output_dir}/model_irregular.csv', index=False, sep=' ', header=False)
        