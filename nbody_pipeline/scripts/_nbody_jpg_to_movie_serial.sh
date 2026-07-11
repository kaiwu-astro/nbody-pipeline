#!/bin/bash

source ~/.bash_aliases
# module use $HOME/.local/easybuild/modules/all && module load ffnvcodec
module load Stages/2025 GCCcore/.13.3.0 CUDA FFmpeg/.7.0.2
# export LD_LIBRARY_PATH=/p/scratch/madnuc/wu13/ffmpeg_cuda/lib:$LD_LIBRARY_PATH && export PATH=/p/scratch/madnuc/wu13/ffmpeg_cuda/bin:$PATH

set -e

# 记录总开始时间
total_start_time=$(date +%s)

make_ffmpeg_list() {
    # example:
    #     many files with name like
    #       simu_1m_5hb_ttot_33.375_mass.jpg
    #       simu_1m_5hb_ttot_6.0_mass.jpg
    #     number is the 5th field, then do
    #     ls *.jpg | make_ffmpeg_list -k 5 > list.txt
    # output:
    #     stdout
    local key=2 # default value for -k
    while getopts k: flag
    do
        case "${flag}" in
            k) key=${OPTARG};;
        esac
    done
    shift $((OPTIND -1))

    sort -t_ -k${key} -n | sed "s/^/file '/" | sed "s/$/'/"
}

# 根据文件模式确定比特率
get_quality_params() {
    local pattern=$1
    case "$pattern" in
        "_x1_vs_x2.jpg") echo "-b:v 5M" ;;
        # "_mass_vs_distance_loglog.jpg") echo "1M" ;;
        # "_CMD.jpg") echo "2M" ;;
        # "_L_vs_Teff_loglog.jpg") echo "1M" ;;
        # "_a_vs_primary_mass_loglog.jpg") echo "1M" ;;
        # "_a_vs_primary_mass_loglog_compact_objects_only.jpg") echo "1M" ;;
        # "_mass_ratio_vs_primary_mass_loglog.jpg") echo "1M" ;;
        # "_mass_ratio_vs_primary_mass_loglog_compact_objects_only.jpg") echo "1M" ;;
        # "_ecc_vs_a.jpg") echo "1M" ;;
        # "_ecc_vs_a_compact_objects_only.jpg") echo "1M" ;;
        # "_ebind_vs_a_loglog.jpg") echo "1M" ;;
        *) echo "-rc constqp -qp 43" ;; # 默认
    esac
}

simu_name_patterns=(
    "_0sb"
    "20sb"
    "60sb"
)

plot_patterns=(
    "_x1_vs_x2.jpg"
    "_mass_vs_distance_loglog.jpg"
    "_CMD.jpg"
    "_L_vs_Teff_loglog.jpg"
    "_a_vs_primary_mass_loglog.jpg"
    "_a_vs_primary_mass_loglog_compact_objects_only.jpg"
    "_mass_ratio_vs_primary_mass_loglog.jpg"
    "_mass_ratio_vs_primary_mass_loglog_compact_objects_only.jpg"
    "_ecc_vs_a.jpg"
    "_ecc_vs_a_compact_objects_only.jpg"
    "_ecc_vs_a_loglog_compact_objects_only.jpg"
    "_ebind_vs_a_loglog.jpg"
    "_ebind_vs_a_loglog_compact_objects_only.jpg"
    "_taugw_vs_a_compact_objects_only.jpg"
    "_mtot_vs_distance_loglog.jpg"
    "_mtot_vs_distance_loglog_compact_objects_only.jpg"
)

n_threads=20

cd ~/scratch/plot/jpg/

declare -a durations_info=() # Initialize an array to store duration info

for simu_name_pattern in "${simu_name_patterns[@]}"; do
    for plot_pattern in "${plot_patterns[@]}"; do
        start_time=$(date +%s) # 记录视频开始时间
        quality_params=$(get_quality_params "$plot_pattern")
        echo "======================================================"
        echo "Making video for $simu_name_pattern $plot_pattern with $quality_params"
        echo "======================================================"
        list_file="$simu_name_pattern$plot_pattern.txt"
        ls *${simu_name_pattern}*${plot_pattern} | make_ffmpeg_list -k 7 > $list_file
        # output name = simu_name_pattern + plot_pattern exclude asterisk
        output_name=${simu_name_pattern}${plot_pattern}.mp4
        rm -f $output_name
        # ffmpeg -threads $n_threads -r 30 -f concat -safe 0 -i $list_file -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" $quality_params -c:v hevc -an -pix_fmt yuv420p -tag:v hvc1 -preset fast -movflags +faststart $output_name # pure cpu encoding (no gpu)
        ffmpeg -y -hwaccel cuda -r 30 -f concat -safe 0 -i $list_file -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" $quality_params -c:v hevc_nvenc -an -pix_fmt yuv420p -tag:v hvc1 -preset p1 -movflags +faststart $output_name # cuda support, reshape strange figsize
        end_time=$(date +%s) # 记录视频结束时间
        duration=$((end_time - start_time)) # 计算持续时间
        durations_info+=("Time taken for $output_name: $duration seconds")
    done
done

# Print all duration information
for info in "${durations_info[@]}"; do
    echo "$info"
done

# 记录总结束时间
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
echo "Total script execution time: $total_duration seconds"
