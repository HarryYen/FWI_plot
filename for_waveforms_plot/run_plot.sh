#!/bin/bash

start=$SECONDS
multi_num=3

work_dir=/home/harry/Work/FWI_result/output_from_ADJOINT_TOMO
evt_file=$work_dir/../rmt_g10.txt
total_lines=`cat $evt_file | wc -l`
lines_per_file=$((total_lines / $multi_num))


rm tmp_*

split -l $lines_per_file $evt_file tmp_
file_list=`ls tmp*`
for file in $file_list;
do
echo 'nohup python wav_plot_parallel.py '"'$file'"' ' | sh &
done
wait

end=$SECONDS
elapsed_time=$((end_time-start_time))
echo "Command took $elapsed_time seconds" > computation_time.txt
