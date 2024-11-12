import numpy as np
import pandas as pd
import pygmt
import sys
import os

def find_minmax(df):
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    dep_min, dep_max = df['dep'].min(), df['dep'].max()
    
    return lon_min, lon_max, lat_min, lat_max, dep_min, dep_max

def check_and_create_dir(model_num, output_dir):
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')
    else:
        pass

def writing_output(output_dir, model_num, abs_final_arr, pert_final_arr, lon_arr, lat_arr, dep_arr, lon_spacing, lat_spacing, dep_spacing):
    lon_min, lon_max = lon_arr[0], lon_arr[-1]
    lat_min, lat_max = lat_arr[0], lat_arr[-1]
    dep_min, dep_max = dep_arr[0], dep_arr[-1]
    nx, ny, nz = len(lon_arr), len(lat_arr), len(dep_arr)
    vp_min, vp_max = np.nanmin(abs_final_arr[0]), np.nanmax(abs_final_arr[0])
    vs_min, vs_max = np.nanmin(abs_final_arr[1]), np.nanmax(abs_final_arr[1])
    rho_min, rho_max = np.nanmin(abs_final_arr[2]), np.nanmax(abs_final_arr[2])
    dvp_min, dvp_max = np.nanmin(pert_final_arr[0]), np.nanmax(pert_final_arr[0])
    dvs_min, dvs_max = np.nanmin(pert_final_arr[1]), np.nanmax(pert_final_arr[1])
    drho_min, drho_max = np.nanmin(pert_final_arr[2]), np.nanmax(pert_final_arr[2])
    
    with open(f'{output_dir}/model_all.xyz', "w") as f:
        f.write(f'{lon_min:.3f}  {lat_min:.3f}  {dep_min:.3f}  {lon_max:.3f}  {lat_max:.3f}  {dep_max:.3f}\n')
        f.write(f'{lon_spacing:.3f}      {lat_spacing:.3f}      {dep_spacing:.3f}\n')
        f.write(f'{nx:5d}      {ny:5d}      {nz:5d}\n')
        f.write(f'{vp_min:10.3f} {vp_max:10.3f} {vs_min:10.3f} {vs_max:10.3f} {rho_min:10.3f} {rho_max:10.3f}\n')
        f.write(f'{dvp_min:10.3f} {dvp_max:10.3f} {dvs_min:10.3f} {dvs_max:10.3f} {drho_min:10.3f} {drho_max:10.3f}\n')

        for kk in range(len(dep_arr)):
            for jj in range(len(lat_arr)):
                for ii in range(len(lon_arr)):
                    lon_tmp, lat_tmp, dep_tmp = lon_arr[ii], lat_arr[jj], dep_arr[kk]
                    vp_tmp, vs_tmp, rho_tmp = abs_final_arr[0][kk][jj][ii], abs_final_arr[1][kk][jj][ii], abs_final_arr[2][kk][jj][ii]
                    dvp_tmp, dvs_tmp, drho_tmp = pert_final_arr[0][kk][jj][ii], pert_final_arr[1][kk][jj][ii], pert_final_arr[2][kk][jj][ii]
                    f.write(f'{lon_tmp:10.3f} {lat_tmp:10.3f} {dep_tmp:10.3f} {vp_tmp:9.3f} {vs_tmp:9.3f} {rho_tmp:9.3f} {dvp_tmp:9.3f} {dvs_tmp:9.3f} {drho_tmp:9.3f}\n')


scalar_list = ['vp', 'vs', 'rho']
x_spacing, y_spacing = 0.08, 0.08
z_spacing = 5.
args = sys.argv[1]
model_n = int(args)
model_num = f'm{model_n:03d}'

print(f'===========Model:{model_num}==============')
csv_dir = f'/home/harry/Work/FWI_result/output_final/{model_num}'
input_file = f'{csv_dir}/model_irregular.csv'
output_dir = csv_dir

xyz_df = pd.read_csv(input_file, sep='\s+', 
                        names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax(xyz_df)
dep_arr = np.arange(0, dep_max+z_spacing, z_spacing)
grd_range = [lon_min, lon_max, lat_min, lat_max]

pert_final_list = []
abs_final_list = []
for scalar in scalar_list:
    print(f'=====> scalar:{scalar}')
    selected_col = ['lon', 'lat', scalar]

    pert_stacked_by_dep_list = []
    abs_stacked_by_dep_list = []

    for dep in dep_arr:
        print(f'==>depth:{dep:.1f}km')
        
        selected_df = xyz_df[(xyz_df.dep <= dep+z_spacing/2) & (xyz_df.dep >= dep-z_spacing/2)][selected_col]
        # selected_df = xyz_df[(xyz_df.dep <= dep+1) & (xyz_df.dep >= dep-2)][selected_col]
        grd = pygmt.xyz2grd(data=selected_df, 
                            # outgrid='tmp.grd', 
                            region=grd_range, 
                            spacing=[x_spacing, y_spacing],
                            verbose = 'q',
                            )

        lon_arr = np.array(grd.x)
        lat_arr = np.array(grd.y)
        scalar_arr = np.array(grd)

        average_scalar = np.nanmean(scalar_arr)
        # average_scalar = grd.sel({'x': lon_arr[1], 'y': lat_arr[lat_arr.size//2]}).item()
        # print(dep, average_scalar)
        perturbation_scalar = scalar_arr/average_scalar
        
        abs_stacked_by_dep_list.append(scalar_arr)
        pert_stacked_by_dep_list.append(perturbation_scalar)

    abs_stacked_by_dep_arr = np.stack(abs_stacked_by_dep_list, axis=0)
    pert_stacked_by_dep_arr = np.stack(pert_stacked_by_dep_list, axis=0)

    abs_final_list.append(abs_stacked_by_dep_arr)
    pert_final_list.append(pert_stacked_by_dep_arr)

abs_final_arr = np.stack(abs_final_list, axis=0)
pert_final_arr = np.stack(pert_final_list, axis=0)


check_and_create_dir(model_num, output_dir)
writing_output(output_dir = output_dir, model_num = model_num, abs_final_arr = abs_final_arr, 
            pert_final_arr = pert_final_arr, 
            lon_arr = lon_arr, lat_arr = lat_arr, dep_arr = dep_arr, 
            lon_spacing = x_spacing, lat_spacing = y_spacing, dep_spacing = z_spacing)
