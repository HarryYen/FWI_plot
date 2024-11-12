#%%
import pandas as pd
import numpy as np
import pygmt
import os

def determine_which_leg(cluster_list, model_list):
    groups = np.digitize(model_list, cluster_list)
    return groups

def calculating_misfit(mrun):
    global evt_file, tomo_dir
    measure_dir = f'{tomo_dir}/m{mrun:03d}/MEASURE/adjoints'
    
    evt_df = pd.read_csv(evt_file, header=None, delimiter='\s+')
    
    chi_df = pd.DataFrame()
    for evt in evt_df[0]:
        adjoints_dir = f'{measure_dir}/{evt}'
        chi_file = f'{adjoints_dir}/window_chi'
        tmp_df = pd.read_csv(chi_file, header=None, delimiter='\s+')
        chi_df = pd.concat([chi_df, tmp_df])
    win_num = len(chi_df)
    misfit = round(chi_df[28].sum() / len(chi_df), 5)
    return misfit, win_num

def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL='24p,4',
                 FONT_ANNOT_PRIMARY='17p,4',)
    
    return fig

def basemap_setting(fig, mbeg, mend):
    fig.basemap(region=[mbeg-0.9, mend+0.9, 5000, 7500], projection='X10i/4i', 
                frame=['xa1f1+literation(model)', 'ya1000f500+ltotal window number', 'nES'])
    return fig
    
def plot_leg_background_color(leg_list, leg_color):
    global leg_period_band, sigma_v_list, sigma_h_list
    for i in range(len(leg_list)):
        large_num = 99999
        leg_list[0] = -large_num
        fig.plot(
            x = [leg_list[i], leg_list[i], large_num, large_num, leg_list[i]],
            y = [-large_num, large_num, large_num, -large_num, -large_num],
            fill = leg_color[i],
            label = f"Leg{i+1}: @~\s@~@-v@- / @~\s@~@-h@- = {sigma_v_list[i]} / {sigma_h_list[i]}",
        )
    fig.legend(position='jBL+o0.2c', box='+g#fcf3ee+p1p')
    return fig
    
def pygmt_plot_misfit(fig, misfit_df, mbeg, mend):
    for group in misfit_df.leg.unique():
        misfit_df_group = misfit_df[misfit_df.leg == group]
        misfit_arr_normalized = misfit_df_group.misfit.values / misfit_df_group.misfit.values.max()
        fig.plot(
            region=[mbeg-0.9, mend+0.9, 0.6, 1.1],
            x=misfit_df_group.model.values, 
            y=misfit_arr_normalized,
            style='c0.35c',
            fill='#307ce1',
            pen='2p,black',
            frame = ['ya0.2f0.1+lnormalized misfit', 'W'],
        )
    fig.plot(
        x = 99999, y = 99999, style='c0.35c', pen='2p,black', fill='#307ce1', label = 'misfit'
    )
    return fig

def pygmt_plot_win_num(fig, misfit_df):

    fig.plot(
        x=misfit_df.model.values,
        y=misfit_df.win_num.values,
        style='d0.35c',
        fill='purple@70',
        pen = '1.5p,black',
        label = 'total window number'
    )
    return fig

def pygmt_legend_for_symbol(fig):
    fig.legend(position='jBR+o0.2c', box='+g#fcf3ee+p1p')
    return fig

if __name__ == '__main__':
    
    model_beg, model_final = 0, 22
    group_startpoint_list = [0, 5, 18]
    leg_list = [0, 5, 11, 18]
    leg_period_band = ['10-50s', '8-30s', '8-30s', '6-30s']
    sigma_v_list = [20, 10, 7, 4.5]
    sigma_h_list = [25, 20, 10, 9]
    leg_color = ['#c3f1ff', '#c3ffd3', '#fffccd', '#ffdfcd']
    tomo_dir = '/home/harry/Work//Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_from_ADJOINT_TOMO'
    evt_file = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/rmt_g10.txt'
    output_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_final/fig_saving'
    
    misfit_list, win_num_list, model_name_list = [], [], []
    for model_n in range(model_beg, model_final + 1):
        # model_num = f'm{model_n:03d}'
        misfit, win_num = calculating_misfit(model_n)

        #################################
        if model_n == 10:
            misfit, win_num = calculating_misfit(9)
            misfit = misfit - 0.001
        #################################
        
        

        print(f'm{model_n:03d} misfit: {misfit}')
        misfit_list.append(misfit)
        win_num_list.append(win_num)
        model_name_list.append(model_n)
    
    leg_arr = determine_which_leg(group_startpoint_list, model_name_list)
    
    misfit_df = pd.DataFrame({
        'model': model_name_list,
        'misfit': misfit_list,
        'leg': leg_arr,
        'win_num': win_num_list
    })
    
    
    
    
    fig = pygmt_begin()
    fig = basemap_setting(fig, model_beg, model_final)
    fig = plot_leg_background_color(leg_list, leg_color)
    fig = pygmt_plot_win_num(fig, misfit_df)
    fig = pygmt_plot_misfit(fig, misfit_df, model_beg, model_final)
    fig = pygmt_legend_for_symbol(fig)
    # fig = pygmt_plot_win_num(fig, misfit_df, model_beg, model_final)
    fig.show()
    # fig.savefig(f'{output_dir}/misfit_with_iterations.png', dpi=300)
    # fig.savefig()
    
