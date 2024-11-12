'''
The runner code for processing the results and plotting the models.
'''

import subprocess
import os

if __name__ == "__main__":
    
    model_beg, model_end = 0, 0
    pre_processing = True
    plot_pert = False
    plot_updating = False
    plot_misfit = False
            
    current_dir = os.getcwd()
    for model_num in range(model_beg, model_end+1): 
        if pre_processing:
            os.chdir(f'{current_dir}/from_vtk_to_csv')
            subprocess.run(['python','from_vtk_to_regular_table.py',f'{model_num}'])
            # subprocess.run(['python','01_vtk2xyz_irregular.py',f'{model_num}'])
            # subprocess.run(['python','02_xyzv2xyzdv.py',f'{model_num}'])
            
        if plot_pert:
            os.chdir(f'{current_dir}/for_plotting_pert_model')
            subprocess.run(['python', 'model_plot_3x3_pert.py',f'{model_num}'])
        
        if plot_updating:
            if model_num != 0:
                os.chdir(f'{current_dir}/for_plotting_updating_amount')
                subprocess.run(['python', 'plotting_updating_model_3x3.py',f'{model_num-1}',f'{model_num}'])
        
        if plot_misfit:
            os.chdir(f'{current_dir}/for_plotting_misfit_kernel')
            subprocess.run(['python', 'vtk2csv.py',f'{model_num}'])
            subprocess.run(['python', 'plotting_kernel_3x3.py',f'{model_num}'])
        


        
    # os.chdir(f'{current_dir}/for_plotting_updating_amount')
    # subprocess.run(['python', 'plotting_updating_model_3x3.py',f'{model_beg}',f'{model_end}'])