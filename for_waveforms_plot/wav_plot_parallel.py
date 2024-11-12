import pygmt
import obspy as obs
from obspy import read 
import sys
import numpy as np
import pandas as pd
import glob
import os
import multiprocessing as mp
import sys
from importlib import reload



def grab_windows_info(work_dir, evt, model_index, comp, sta):
    window_file = f'{work_dir}/m{model_index:03d}/MEASURE/adjoints/{evt}/window_index'
    df = pd.read_csv(window_file, delimiter='\s+', header=None, 
                     names=['net', 'sta', 'comp', 'u1', 'u2', 'u3', 'u4', 't1', 't2'])
    df = df[(df['sta'] == sta) & (df['comp'] == f'BH{comp}')]
    return df

def make_dir(save_dir, evt):
    new_save_dir = f'{save_dir}/{evt}'
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    return new_save_dir


def plot(evt, comp_list, comp_color, mbeg, mend, save_dir, work_dir):
    import pygmt
    reload(pygmt)
    get_sta = glob.glob(f'{work_dir}/m{mbeg:03d}/OBS/{evt}/*')
    sta_list = [sta.split('/')[-1].split('.')[0] for sta in get_sta]
    sta_list_filter = list(set(sta_list))
    new_save_dir = make_dir(save_dir, evt)
    print(evt)
    for sta in sta_list_filter:
        sta = sta.split('/')[-1].split('.')[0]
        print(sta)
        fig = pygmt.Figure()
        pygmt.config(FONT_HEADING="22p,Helvetica-Bold,black")
        with fig.subplot(nrows=mend+1, ncols=3, 
                        figsize=("30c", "46c"),
                        autolabel=0,
                        title=f'{sta}, {evt}',
                        margins=["0.1c", "0.1c"],
			            sharex=True,
        ):
            for comp_ii, comp in enumerate(comp_list):
                print(f'==={evt}, {sta}, {comp}===')
            
                obs_list = []
                syn_list = []
                for num in range(mbeg, mend+1):
                    model_num = f'm{num:03d}'
                    syn_sac = f'{work_dir}/{model_num}/SYN/{evt}/{sta}.TW.BH{comp}.semv.sac.tomo'
                    obs_sac = f'{work_dir}/{model_num}/OBS/{evt}/{sta}.TW.BH{comp}.sac.tomo'
                    print(obs_sac)
                    try:
                        stream_syn = read(syn_sac)
                        stream_obs = read(obs_sac)
                        st_syn = stream_syn.copy()
                        st_obs = stream_obs.copy()
                        tr_syn = st_syn[0]
                        tr_obs = st_obs[0]
                        obs_list.append(tr_obs)
                        syn_list.append(tr_syn)
                        
                    except FileNotFoundError:
                        tr_syn = 'NaN'
                        tr_obs = 'NaN'	
                        obs_list.append(tr_obs)
                        syn_list.append(tr_syn)
                        print(f'Warning: {evt}/{sta}.TW.BH{comp} has some problems...')
                        pass

                if len(obs_list) != len(syn_list):
                    sys.exit("The number of obs and syn are not equal.")


                total_model_num = len(obs_list)
                for ii in range(total_model_num):
                    if ii == total_model_num - 1:
                        frame = ["newS"]
                    else:
                        frame = ["news"]
               
                    jj = comp_ii
                    index = ii * len(comp_list) + jj
                    model_num_tmp = mbeg + ii
                    label = f"m{model_num_tmp:02d}"
                    tr_syn = syn_list[ii]
                    tr_obs = obs_list[ii]
                    if tr_syn == 'NaN':
                        continue
                    max_syn = max(np.abs(tr_syn.data))
                    max_obs = max(np.abs(tr_obs.data))
                    max_val = max(max_syn, max_obs) * 1.02
                    
                    window_df = grab_windows_info(work_dir, evt, ii, comp, sta)
                    with fig.set_panel(panel=index, fixedlabel=label):
                        fig.basemap(region=[-25, 120, -max_val, max_val], frame=frame)
                        for plot_windows_ii in range(len(window_df)):
                            t1, t2 = window_df.iloc[plot_windows_ii]['t1'], window_df.iloc[plot_windows_ii]['t2']
                            fig.plot(x=[t1, t2, t2, t1, t1], y=[-99999, -99999, 99999, 99999, -99999], pen='1p,black', fill='pink@50')
                        fig.plot(x = tr_obs.times()+tr_obs.stats.sac.b, y = tr_obs.data, pen=f"2.0p,black")
                        fig.plot(x = tr_syn.times()+tr_syn.stats.sac.b, y = tr_syn.data, pen=f"1.7p,{comp_color[jj]}")
                        

                                
        fig.savefig(f"{new_save_dir}/{sta}.png")


if __name__ == '__main__':
    
    mbeg, mend = 10, 14
    comp_list = ['Z','N','E']
    comp_color = ['red', 'purple', 'green']
    work_dir = '/home/harry/Work/FWI_result/output_from_ADJOINT_TOMO'
    save_dir = f'{work_dir}/../waveform_view'
    event_file = sys.argv[1]
    event_df = pd.read_csv(event_file, delimiter='\s+', header=None)
    event_list = list(event_df.iloc[:,0])

    mp_list = [(evt, comp_list, comp_color, mbeg, mend, save_dir, work_dir) for evt in event_list]
    for mp_args in mp_list:
        print(mp_args)
        plot(*mp_args)
    # mp_args = mp_list[0]
    # plot(*mp_args)    
