#%%
import pygmt
import pandas as pd
import numpy as np
from pygmt.datasets import load_earth_relief
import sys
import math
import os

def find_minmax(df):
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    dep_min, dep_max = df['dep'].min(), df['dep'].max()
    
    return lon_min, lon_max, lat_min, lat_max, dep_min, dep_max

def find_dxdydz(input_file):
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            if index == 2:
                values = line.strip().split()
                nx = int(values[0])
                ny = int(values[1])
                nz = int(values[2])
    return nx, ny, nz

def check_and_create_dir(model_num, output_dir):
    output_model_dir = f'{output_dir}/fig_saving/model_fig_updated_3x3'
    if not os.path.exists(f'{output_model_dir}'):
        os.makedirs(f'{output_model_dir}')
    return output_model_dir

args = sys.argv[1:3]
m_beg = int(args[0])
m_end = int(args[1])

map_region = [119, 123, 21, 26.]

input_dir = '/home/harry/Work/FWI_result/output_final'
scalar_list = ['vp','vs','rho']
x_spacing, y_spacing = 0.08, 0.08
z_spacing = 0.5
depth_list = [5, 10, 15, 20, 30, 50, 80, 120, 150]
# range_for_colormap = [-6, 6, 0.5] # [min, max, interval]

model_num_list = [f'm{m_beg:03d}', f'm{m_end:03d}']

df_list = []
for model_num in model_num_list:
    print(f'===========Model:{model_num}==============')
    input_file = f'{input_dir}/{model_num}/model_all.xyz'
    nx, ny, nz = find_dxdydz(input_file)
    xyz_df = pd.read_csv(input_file, sep='\s+', skiprows=5, 
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho'])
    df_list.append(xyz_df)
    print(f'=========>{model_num}')

model0_df = df_list[0]
model1_df = df_list[1]  
lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax(model0_df)
for scalar in scalar_list:
    print(f'======>{scalar}')
    model0_arr = np.array(model0_df[scalar])
    model1_arr = np.array(model1_df[scalar])
    model_1_0_diff = model1_arr - model0_arr
    model_1_0_pert = model_1_0_diff / model0_arr * 1E+02
    updating_df = pd.DataFrame(zip(model0_df['lon'], model0_df['lat'], 
                                   model0_df['dep'], model_1_0_pert), 
                               columns=['lon', 'lat', 'dep', scalar])
    # remove the nan in the array
    model_pert_without_nan = model_1_0_pert[~np.isnan(model_1_0_pert)]
    absmax = max(abs(model_pert_without_nan.min()), abs(model_pert_without_nan.max()))
    absmax_ceil = math.ceil(absmax)
    range_for_colormap = [-absmax_ceil, absmax_ceil, absmax_ceil/50]
    print(scalar, range_for_colormap)
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
    pygmt.makecpt(cmap='jet', series=range_for_colormap, reverse=True)
    with fig.subplot(nrows=3, ncols=3, figsize=("17c", "17c"), margins='0.005c', frame=['a', 'WSne']):
        for i in range(3): 
            for j in range(3): 
                index = i * 3 + j
                dep = depth_list[index]
                print(f'====>{dep}km')
                grd_range = [lon_min, lon_max, lat_min, lat_max]
                selected_col = ['lon', 'lat', scalar]
                selected_df = updating_df[(updating_df.dep <= dep+z_spacing/2) & (updating_df.dep >= dep-z_spacing/2)][selected_col]
                
                pygmt.xyz2grd(data=selected_df, 
                            outgrid='tmp.grd', 
                            region=grd_range, 
                            spacing=f'{nx}+n/{ny}+n',
                            verbose='q')
                pygmt.grdsample(grid='tmp.grd', spacing=0.02, 
                                region=grd_range, outgrid='tmp_fine.grd',
                                verbose='q')
                with fig.set_panel(panel=index):  # sets the current panel
                    fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M?', frame=True)
                    fig.coast(shorelines=True)
                    fig.text(text=f"dep: {depth_list[index]:3d}km", font="12p,Helvetica-Bold", position="BR", frame=True)

        # fig.plot(x = selected_df.lon, y = selected_df.lat, style='c0.02c', fill='black', pen='black')
        output_model_dir = check_and_create_dir(model_num, input_dir)

        fig.colorbar(frame=f'a{absmax_ceil}f{absmax_ceil/2}+l"{scalar} updated(%)"')
        fig.savefig(f'{output_model_dir}/{scalar}_m{m_beg}_m{m_end}.png', dpi=300)

# %%
