#%%
from obspy import read
import numpy as np
import pandas as pd
import glob
import os
import sys
import pygmt

def create_meca_dataframe(file):
    df = pd.read_csv(file, sep="\s+", header=None, usecols=range(20), names=[
        'formatted_datetime', 'date', 'time', 'long', 'lat', 'depth', 
        'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
        'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'
    ])
    return df

def modify_meca_format(df, evt):
    
    df_sort = df[df.formatted_datetime == float(evt)]
    df_meca = df_sort[['long', 'lat', 'depth', 'strike1', 'dip1', 'rake1', 'Mw']]
    # df_meca.columns = ['longitude','latitude', 'depth',
                    # 'strike', 'dip', 'rake', 'magnitude']
    meca_info = np.array(list(df_meca.iloc[0]))

    return meca_info
    
        

def generate_evt_win_dict(ref_win_dir, model_ref):
    '''
    read window index file from reference model
    here we output 2 dictonaries: evt_dict and win_dict_evt
    1. evt_dict: {evt: win_dict_evt}
    2. win_dict_evt: {(sta, comp): [(t1, t2)]}
    '''
    evt_dict = {}
    for dir in glob.glob(f'{ref_win_dir}/*'):
        evt = dir.split('/')[-1]
        win_file = f'{dir}/window_index'
        df = pd.read_csv(win_file, delimiter='\s+', header=None, 
                         names=['net', 'sta', 'comp', 'u1', 'u2', 'u3', 'u4', 't1', 't2'])
       
        win_dict_evt = {
            (sta, comp): list(zip(group['t1'], group['t2']))
            for (sta, comp), group in df.groupby(['sta', 'comp'])
        }
        
        evt_dict[evt] = win_dict_evt
    
    return evt_dict

def generate_sta_dataframe(obs_wav_dir, win_dict_evt):
            
    sta_list, azi_list = [], []
    stlo_list, stla_list = [], []
    evlo_list, evla_list = [], []
    for (sta, comp), win_list in win_dict_evt.items():
        try:
            obs_sac = glob.glob(f'{obs_wav_dir}/{sta}.TW.{comp}.sac.tomo')[0]
        except IndexError:
            print(f"{obs_wav_dir}/{sta}.TW.{comp}.sac.tomo' doesn't exist!")
        
        stream = read(obs_sac)
        st = stream.copy()
        tr = st[0]
        azi = tr.stats.sac.az
        stlo, stla = tr.stats.sac.stlo, tr.stats.sac.stla
        evlo, evla = tr.stats.sac.evlo, tr.stats.sac.evla
        
        sta_list.append(sta)
        azi_list.append(azi)
        stlo_list.append(stlo)
        stla_list.append(stla)
        evlo_list.append(evlo)
        evla_list.append(evla)
    
    sta_azi_df = pd.DataFrame({'sta': sta_list, 'azi': azi_list, 'stlo': stlo_list, 'stla': stla_list, 'evlo': evlo_list, 'evla': evla_list})
    sta_azi_df = sta_azi_df.drop_duplicates()
    sta_azi_df = sta_azi_df.sort_values(by='azi')
    
    return sta_azi_df

def chunck_dataframe(df, chunksize=10):
    df_splits = [df.iloc[i: i + chunksize] for i in range(0, len(df), chunksize)]
    return df_splits

def Pygmt_config():
    pygmt.config(MAP_FRAME_TYPE="plain",
                 FORMAT_GEO_MAP="ddd.x",
                 FONT_ANNOT_SECONDARY = '24p,0',
                 FONT_ANNOT_PRIMARY = '24p, 0',
                 FONT_LABEL = '24p, 0')
    

def plot_map(map_region, sta_df, meca_info):
    fig = pygmt.Figure()
    # fig.basemap(region=map_region, projection="M8i", frame=['a2f1', 'WSne'])
    fig.coast(shorelines=True, region=map_region, projection="M8i", frame=['a2f1', 'WSne'])
    
    x_coords = []
    y_coords = []
    sta_list = []
    for line in sta_df.itertuples():
        x_coords.extend([line.stlo, line.evlo, np.nan])
        y_coords.extend([line.stla, line.evla, np.nan])
        sta_list.append(line.sta)
        
    fig.plot(x=x_coords, y=y_coords, pen='0.8p,black')    
    fig.plot(x = sta_df.stlo, y = sta_df.stla, style='t1c', fill='blue', pen='black')
    
    pygmt.makecpt(cmap='jet', series=[0, 200], reverse=True)
    
    fig.meca(
        spec = meca_info,
        convention = 'aki',
        cmap = True,
        scale = "1.6c"
    )
    
    fig.text(x=sta_df.stlo, y=sta_df.stla - 0.15, text=sta_df.sta, 
             font='18p,Helvetica-Bold', justify='CM', fill='#ffffaa')
    
    fig.colorbar(cmap = True, position = 'x0.5c/0.5c+w7c/0.6c+m+h', frame = ['a20f10','+L"Depth (km)"'])  
    
    return fig

# def grab_wavs_info(evt_df, win_dict):
#     for sta in evt_df.sta:
        
        # sorted_dict = {key: value for key, value in win_dict.items() if key[0] == sta}
        
def read_sac(sac):
    try:
        stream = read(sac)
        st = stream.copy()
        tr = st[0]
        wav = tr.data
        
        times = tr.times()
    except FileNotFoundError:
        wav = np.nan
        times = np.nan
    return wav, times

def plot_waveforms(fig, evt, waveform_time_range, chunk_df, win_dict_evt, ref_syn_wav_dir, final_syn_wav_dir, ref_wav_dir):
    global wav_start_time
    
    fig.shift_origin(xshift="22c")
    with fig.subplot(nrows=10, ncols=3, subsize=('15c', '5c'), margins=["0.1c", "0.1c"], 
                     sharex='b', sharey='r'):
        for i in range(10):
            sta_info = chunk_df.iloc[i]
            sta = sta_info.sta          
            for j, channel in enumerate(['BHZ', 'BHN', 'BHE']): 
                index = i * 3 + j  
                
                # get waveform
                ref_syn_sac = f'{ref_syn_wav_dir}/{evt}/{sta}.TW.{channel}.semv.sac.tomo'
                final_syn_sac = f'{final_syn_wav_dir}/{evt}/{sta}.TW.{channel}.semv.sac.tomo'
                ref_data_sac = f'{ref_wav_dir}/{evt}/{sta}.TW.{channel}.sac.tomo'
                wav_data, data_time           = read_sac(ref_data_sac)
                wav_syn_ref, ref_syn_time     = read_sac(ref_syn_sac)
                wav_syn_final, final_syn_time = read_sac(final_syn_sac)
                max_val = np.max(np.abs(np.hstack([wav_data, wav_syn_ref, wav_syn_final])))
                
                # normalize
                wav_data = wav_data / max_val
                wav_syn_ref = wav_syn_ref / max_val
                wav_syn_final = wav_syn_final / max_val
                
                with fig.set_panel(panel=index):
                    fig.basemap(region = [waveform_time_range[0], waveform_time_range[1], -1.05, 1.05], 
                                frame = ['xa20f10', 'yf1'], projection = 'X?')
                    
                    # get window info
                    try:
                        win_values = win_dict_evt[(sta, channel)]
                        for win in win_values:
                            t1, t2 = win
                            fig.plot(x=[t1, t2, t2, t1, t1], y=[-9999, -9999, 9999, 9999, -9999], pen='1p,black', fill='pink@50')
                    except KeyError:
                        pass
                    fig.plot(x = data_time + wav_start_time, y = wav_data, pen='1.8p,black')
                    fig.plot(x = ref_syn_time + wav_start_time, y = wav_syn_ref, pen='1.8p,blue,--')
                    fig.plot(x = final_syn_time + wav_start_time, y = wav_syn_final, pen='1.8p,red')
                      
                    
    return fig
        
def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    
    # ---------------- PARAMETER -----------------#
    model_ref, model_final = 0, 4
    map_region = [119, 123, 21, 26]
    result_dir = '/home/harry/Work/FWI_result/output_from_ADJOINT_TOMO'
    evt_file = '/home/harry/Work/FWI_result/rmt_g10.txt'
    waveform_time_range = [0, 120]
    wav_start_time = -30.
    output_dir = '/home/harry/Work/FWI_result/waveform_with_azi'
    # --------------------------------------------#
    
    ensure_directory_exists(output_dir)
    
    evt_meca_df = create_meca_dataframe(evt_file)
    
    model_ref = f'm{model_ref:03d}'
    model_final = f'm{model_final:03d}'
    ref_win_dir = f'{result_dir}/{model_ref}/MEASURE/adjoints'
    ref_wav_dir = f'{result_dir}/{model_ref}/OBS'
    ref_syn_wav_dir = f'{result_dir}/{model_ref}/SYN'
    final_syn_wav_dir = f'{result_dir}/{model_final}/SYN'
    
    chunksize = 10
    
    evt_dict = generate_evt_win_dict(ref_win_dir, model_ref)
    evt_list = list(evt_dict.keys())
    # SET GMT CONFIG
    Pygmt_config()
    
    for evt, win_dict_evt in evt_dict.items():
        obs_wav_dir = f'{ref_wav_dir}/{evt}'
        sta_azi_df = generate_sta_dataframe(obs_wav_dir, win_dict_evt)
        chunk_list = chunck_dataframe(df = sta_azi_df, chunksize=chunksize)
        
        meca_info = modify_meca_format(evt_meca_df, evt)
        for ii, chunk_df in enumerate(chunk_list):
            fig = plot_map(map_region, chunk_df, meca_info)
            fig = plot_waveforms(fig, evt, waveform_time_range, chunk_df, win_dict_evt, 
                           ref_syn_wav_dir, final_syn_wav_dir, ref_wav_dir)
            fig.savefig(f'{output_dir}/{evt}_{ii:02d}.png')
            sys.exit()
 
# %%
