#%%
from scipy.interpolate import RegularGridInterpolator
import pygmt
import pandas as pd
import numpy as np
import sys
import os
import xarray as xr

def find_minmax(input_file):
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            if index == 0:
                values = line.strip().split()
                lon_min = float(values[0])
                lat_min = float(values[1])
                dep_min = float(values[2])
                lon_max = float(values[3])
                lat_max = float(values[4])
                dep_max = float(values[5])
    return lon_min, lon_max, lat_min, lat_max, dep_min, dep_max

def find_nxnynz(input_file):
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            if index == 2:
                values = line.strip().split()
                nx = int(values[0])
                ny = int(values[1])
                nz = int(values[2])
    return nx, ny, nz

def check_and_create_dir(model_num, output_dir):
    output_model_dir = f'{output_dir}/fig_saving/model_fig_3x3'
    if not os.path.exists(f'{output_model_dir}'):
        os.makedirs(f'{output_model_dir}')
    return output_model_dir

def interp_2d_in_specific_dep(lon_uniq, lat_uniq, dep_uniq, arr, target_dep):

    interpolator = RegularGridInterpolator((dep_uniq, lon_uniq, lat_uniq), arr)
    lon_grid, lat_grid = np.meshgrid(lon_uniq, lat_uniq)
    # points = np.array([lon_grid.ravel(), lat_grid.ravel(), np.full(lon_grid.size, target_dep)]).T
    points = np.array([np.full(lon_grid.size, target_dep), lon_grid.ravel(), lat_grid.ravel()]).T
    interpolated_values = interpolator(points)
    df = pd.DataFrame({
    'lon': lon_grid.ravel(),
    'lat': lat_grid.ravel(),
    'scalar': interpolated_values
    })
    
    return df

def to_xarray_2d(data, lon_arr_uniq, lat_arr_uniq, name):
    data_array = xr.DataArray(
    data, 
    dims=['lon', 'lat'],
    coords={'lon': lon_arr_uniq,  
            'lat': lat_arr_uniq},
    name=name 
)
    return data_array

    

map_region = [119.1, 123, 21., 26.]

input_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_final'
scalar_list = ['dvp','dvs','drho']
x_spacing, y_spacing = 0.08, 0.08
depth_list = [6, 10, 15, 20, 30, 50, 80, 120, 150]
vp_range = [-25, 25]
vs_range = [-35, 35]
rho_range = [-20, 20]

args = sys.argv[1]
model_n = int(args)
model_num = f'm{model_n:03d}'

print(f'===========Model:{model_num}==============')
input_file = f'{input_dir}/{model_num}/model.xyz'

nx, ny, nz = find_nxnynz(input_file)
lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax(input_file)

all_arr_flat = np.loadtxt(input_file, skiprows=5)
grid_arr_flat = all_arr_flat[:,0:3]
lon_arr, lat_arr, dep_arr = grid_arr_flat[:,0], grid_arr_flat[:,1], grid_arr_flat[:,2]
lon_arr_uniq, lat_arr_uniq, dep_arr_uniq = np.unique(lon_arr), np.unique(lat_arr), np.unique(dep_arr)
vp_arr_flat, vs_arr_flat, rho_arr_flat = all_arr_flat[:,3], all_arr_flat[:,4], all_arr_flat[:,5]
dvp_arr_flat, dvs_arr_flat, drho_arr_flat = all_arr_flat[:,6], all_arr_flat[:,7], all_arr_flat[:,8]
dvp_arr = dvp_arr_flat.reshape(nz, nx, ny)
dvs_arr = dvs_arr_flat.reshape(nz, nx, ny)
drho_arr = drho_arr_flat.reshape(nz, nx, ny)
grid_arr = grid_arr_flat.reshape(nz, nx, ny, 3)


cpt_range_list = [vp_range, vs_range, rho_range]
model_arr_list = [dvp_arr, dvs_arr, drho_arr]
grd_range = [lon_min, lon_max, lat_min, lat_max]
print(f'=========>{model_num}')
for ii, scalar in enumerate(scalar_list):
    print(f'======>{scalar}')
    fig = pygmt.Figure()
    pygmt.makecpt(cmap='../../cpt_file/Vp_ptb.cpt', series=cpt_range_list[ii], reverse=False)
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
    with fig.subplot(nrows=3, ncols=3, figsize=("17c", "17c"), margins='0.005c', frame=['a', 'WSne']):
        for i in range(3): 
            for j in range(3): 
                index = i * 3 + j
                dep = depth_list[index]
                print(f'====>{dep}km')
                model_arr = model_arr_list[ii]
                selected_df = interp_2d_in_specific_dep(lon_arr_uniq, lat_arr_uniq, dep_arr_uniq, model_arr, dep)

                pygmt.xyz2grd(data=selected_df, 
                            outgrid='tmp.grd', 
                            region=grd_range, 
                            spacing=f'{nx}+n/{ny}+n',
                            verbose='q')
                pygmt.grdsample(grid='tmp.grd', spacing=0.001, 
                                region=grd_range, outgrid='tmp_fine.grd',
                                verbose='q')

                with fig.set_panel(panel=index):  # sets the current panel
                    fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M?', frame=True)
                    fig.coast(shorelines=True)
                    fig.text(text=f"dep: {depth_list[index]:3d}km", font="12p,Helvetica-Bold", position="BR", frame=True)

        # fig.plot(x = selected_df.lon, y = selected_df.lat, style='c0.02c', fill='black', pen='black')
        output_model_dir = check_and_create_dir(model_num, input_dir)

        fig.colorbar(frame=f'a10f10+l"{scalar}"')
        fig.savefig(f'{output_model_dir}/{scalar}_{model_num}.png', dpi=300)



# %%
