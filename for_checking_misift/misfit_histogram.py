#%%
import pandas as pd
import numpy as np
import pygmt
import os
import sys

def generate_misfit_array(evt_arr, model_num):
    global tomo_dir
    
    tt_misfit_list, amp_misfit_list = [], []
    for evt in evt_arr:
        print(evt)
        misfit_dir = f'{tomo_dir}/{model_num}/MEASURE/adjoints/{evt}'
        misfit_file = f'{misfit_dir}/window_chi'
        df = pd.read_csv(misfit_file, sep='\s+', header=None)
    
        tt_misfit_list.append(df[12].values)
        amp_misfit_list.append(df[13].values)
    tt_misfit_arr = np.concatenate(tt_misfit_list)
    amp_misfit_arr = np.concatenate(amp_misfit_list)
    
    return tt_misfit_arr, amp_misfit_arr

def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL='24p',
                 FONT_ANNOT_PRIMARY='15p',)
    
    return fig

def pygmt_plot_histogram_dt(fig, data1, data2, interval, x_range):
    
    num_1, num_2 = len(data1), len(data2)
   # Create histogram for data02 by using the combined data set
    fig.histogram(
        region=[x_range[0], x_range[1], 0, 0],
        projection="X13c",
        frame=["WSne+tMisfit Histogram: dt", "xaf10+ldt(sec)", "yaf500+lCounts"],
        data=data2,
        series=interval,
        fill="skyblue",
        pen="1p,blue,solid",
        histtype=0,
        # The combined data set appears in the final histogram visually
        # as data set data02
        label=f"M_final(N={num_2})",
    )

    # Create histogram for data01
    # It is plotted on top of the histogram for data02
    fig.histogram(
        data=data1,
        series=interval,
        pen="2p,gray,solid",
        fill='orange',
        histtype=0,
        label=f"M_init(N={num_1})",
    )
    
    fig.histogram(
        data=data2,
        series=interval,
        pen="2p,blue,solid",
        histtype=0,
    )
    
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p1p")
    
    
    return fig

def pygmt_plot_histogram_dlnA(fig, data1, data2, interval, x_range):
    
    fig.shift_origin(xshift="16c")
    
    num_1, num_2 = len(data1), len(data2)
   # Create histogram for data02 by using the combined data set
    fig.histogram(
        region=[x_range[0], x_range[1], 0, 0],
        projection="X13c",
        frame=["WSne+tMisfit Histogram: dlnA", "xaf10+ldlnA", "yaf500+lCounts"],
        data=data2,
        series=interval,
        fill="skyblue",
        pen="1p,blue,solid",
        histtype=0,
        # The combined data set appears in the final histogram visually
        # as data set data02
        label=f"M_final(N={num_2})",
    )

    # Create histogram for data01
    # It is plotted on top of the histogram for data02
    fig.histogram(
        data=data1,
        series=interval,
        pen="2p,gray,solid",
        fill='orange',
        histtype=0,
        label=f"M_init(N={num_1})",
    )
    
    fig.histogram(
        data=data2,
        series=interval,
        pen="2p,blue,solid",
        histtype=0,
    )
    
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p1p")
    
    
    return fig

def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == '__main__':
    
    # -----------------PARAMETERS----------------- #
    tomo_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_from_ADJOINT_TOMO'
    event_file = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/rmt_g10.txt'
    out_dir = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_final/fig_saving'
    model_n_list = [18, 22]
    # -------------------------------------------- #
    
    evt_df = pd.read_csv(event_file, sep='\s+', header=None)
    evt_arr = evt_df[0].values

    
    model_num_1 = f'm{model_n_list[0]:03d}'
    model_num_2 = f'm{model_n_list[1]:03d}'
    

    
    tt_misfit_arr1, amp_misfit_arr1 = generate_misfit_array(evt_arr, model_num_1)
    tt_misfit_arr2, amp_misfit_arr2 = generate_misfit_array(evt_arr, model_num_2)
    
    fig = pygmt_begin()
    fig = pygmt_plot_histogram_dt(fig, tt_misfit_arr1, tt_misfit_arr2, 0.6, [-4.5, 4.5])
    # fig = pygmt_plot_histogram_dt(fig, tt_misfit_arr1, tt_misfit_arr2, , [0, 1E-07])
    
    fig = pygmt_plot_histogram_dlnA(fig, amp_misfit_arr1, amp_misfit_arr2, 0.35, [-3, 3])
    
    output_dir = check_output_dir(f'{out_dir}/misfit_histogram')
    output_file = f'{output_dir}/{model_num_1}_{model_num_2}.png'
    # fig.savefig(output_file, dpi=300)
    fig.show()