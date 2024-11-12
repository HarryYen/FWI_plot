#%%
import pandas as pd
import numpy as np
import pygmt
import sys

def pygmt_begin():
    global region_plot
    
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                 FONT_LABEL='24p, 4',
                 FONT_ANNOT_PRIMARY='18p, 4')
    fig.basemap(region=region_plot, projection="M15c", frame=["neWS", "a1f1"])
    fig.coast(shorelines='.5p')
    
    return fig

def plot_sta(fig, sta_df):
    lon_arr = sta_df['lon'].values
    lat_arr = sta_df['lat'].values
    fig.plot(
        x=lon_arr, y=lat_arr, style="t0.4c", fill="#b9eeff", pen = '.5p,black'
    )
    return fig
    

def plot_eq_dirstribution(fig, df_meca):

    
    pygmt.makecpt(cmap='jet', series=[0, 200], reverse=True)
    fig.meca(
        spec = df_meca,
        convention = 'aki',
        cmap = True,
        scale = "0.45c",
        pen = '0.1p,black'
    )

    fig.colorbar(frame=["xa40f20", "y+lDepth(km)"])
    
    return fig

def plot_ray(fig, meca, sta_df):
    sta_lon_arr = sta_df['lon'].values
    sta_lat_arr = sta_df['lat'].values
    evt_lon_arr = meca['longitude'].values
    evt_lat_arr = meca['latitude'].values

    mat_A, mat_B = np.meshgrid(sta_lon_arr, evt_lon_arr)
    lon_comb = np.vstack([mat_A.ravel(), mat_B.ravel()]).T
    lon_flat = lon_comb.flatten()
    
    mat_A, mat_B = np.meshgrid(sta_lat_arr, evt_lat_arr)
    lat_comb = np.vstack([mat_A.ravel(), mat_B.ravel()]).T
    lat_flat = lat_comb.flatten()
    
    insert_positions = np.arange(2, len(lon_comb), 2)
    repeated_lon = np.insert(lon_flat, insert_positions, np.nan)
    repeated_lat = np.insert(lat_flat, insert_positions, np.nan)

    fig.plot(
                x = repeated_lon,
                y = repeated_lat,
                pen='0.01p,black'
            )

    return fig
    



if __name__ == "__main__":
    # --------------------------------------------------------------------------------------#
    
    fig_name = 'ray' 
    evt_file = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/rmt_g10.txt'
    sta_file = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/sta_bb.txt'
    region_plot = [119, 123.5, 21, 26.]
    
    # --------------------------------------------------------------------------------------#

    evt_df = pd.read_csv(evt_file, sep="\s+", header=None, names=[
        'formatted_datetime', 'date', 'time', 'long', 'lat', 'depth', 
        'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
        'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp',
        'u1', 'u2', 'u3', 'u4'
    ])
    
    sta_df = pd.read_csv(sta_file, sep="\s+", header=None, names=[
        'sta', 'lon', 'lat', 'elev'
    ])
    
    sta_df = sta_df[(sta_df['lon'] >= region_plot[0]) & (sta_df['lon'] <= region_plot[1]) &
                    (sta_df['lat'] >= region_plot[2]) & (sta_df['lat'] <= region_plot[3])]
    print(sta_df)

    df_meca = evt_df[['long', 'lat', 'depth', 'strike1', 'dip1', 'rake1', 'Mw']]

    df_meca.columns = ['longitude','latitude', 'depth',
                    'strike', 'dip', 'rake', 'magnitude']
    fig = pygmt_begin()
    
    fig = plot_ray(fig, df_meca, sta_df)
    
    fig = plot_sta(fig, sta_df)
    fig = plot_eq_dirstribution(fig, df_meca)
    
    fig.savefig(f"{fig_name}.png", dpi=300)
    fig.show()


# %%
