#!/bin/bash

set -e

SCRIPT_NAME=$(basename "$0")

show_top_help() {
    cat <<EOF
Usage:
  bash $SCRIPT_NAME
  bash $SCRIPT_NAME create <plot_pattern> [plot_pattern ...]
  bash $SCRIPT_NAME help

Commands:
  create    Create movies for one or more plot patterns.
            Run 'bash $SCRIPT_NAME create help' to list preset patterns.

Without arguments, this script creates movies for all preset plot patterns.
EOF
}

show_create_help() {
    cat <<EOF
Usage:
  bash $SCRIPT_NAME create <plot_pattern> [plot_pattern ...]

Create movies for one or more plot patterns. Patterns are matched directly
against JPG filenames, so custom suffixes are allowed.

Preset plot patterns:
EOF
    for plot_pattern in "${plot_patterns[@]}"; do
        echo "  $plot_pattern"
    done
}

# 记录总开始时间
total_start_time=$(date +%s)

PLOT_DIR="$HOME/scratch/plot/jpg" # jpg图片所在的目录
MAX_PARALLEL_JOBS=15

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

VIDEO_FILTER="setsar=1,pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0,format=yuv420p"

# 根据文件模式确定 GPU 编码质量
get_gpu_quality_params() {
    local pattern=$1
    case "$pattern" in
        "_x1_vs_x2.jpg") echo "-b:v 5M" ;;
        *) echo "-rc constqp -qp 43" ;; # 默认
    esac
}

get_cpu_quality_params() {
    echo "-crf 18"
}

ffmpeg_cuda_available() {
    local has_cuda=0
    local has_hevc_nvenc=0

    if ! ffmpeg -hide_banner -hwaccels >/dev/null 2>&1; then
        return 2
    fi
    if ! ffmpeg -hide_banner -encoders >/dev/null 2>&1; then
        return 2
    fi

    ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -qw cuda && has_cuda=1
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -qw hevc_nvenc && has_hevc_nvenc=1

    [[ $has_cuda -eq 1 && $has_hevc_nvenc -eq 1 ]]
}

ffmpeg_libx265_available() {
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -qw libx265
}

run_ffmpeg_cpu() {
    local list_file="$1"
    local quality_params="$2"
    local output_name="$3"

    if ! ffmpeg_libx265_available; then
        echo "CPU encoder libx265 is not available in this FFmpeg build." >&2
        return 1
    fi

    ffmpeg -y -r 30 -f concat -safe 0 -i "$list_file" -vf "$VIDEO_FILTER" $quality_params -c:v libx265 -an -pix_fmt yuv420p -tag:v hvc1 -preset medium -movflags +faststart "$output_name"
}

run_ffmpeg_cuda() {
    local list_file="$1"
    local quality_params="$2"
    local output_name="$3"

    ffmpeg -y -hwaccel cuda -r 30 -f concat -safe 0 -i "$list_file" -vf "$VIDEO_FILTER" $quality_params -c:v hevc_nvenc -an -pix_fmt yuv420p -tag:v hvc1 -preset p1 -movflags +faststart "$output_name"
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
    # "_allstar_vx_vs_x.jpg"
    "_bin_vx_vs_x.jpg"
    "_bin_vx_vs_x_compact_objects_only.jpg"
    "_a_vs_distance.jpg"
    "_a_vs_distance_compact_objects_only.jpg"
    "_x1_vs_x2_wide_pc.jpg"
    "_orbital_xT_vs_xR_wide_pc.jpg"
    "_orbital_xT_vs_xL_wide_pc.jpg"
    "_x1_vs_x2_highlight_compact_objects.jpg"
    "_x1_vs_x2_highlight_compact_objects_wide_pc.jpg"
)

case "${1:-}" in
    -h|--help|help)
        show_top_help
        exit 0
        ;;
    create)
        shift
        if [[ $# -eq 0 || "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
            show_create_help
            exit 0
        fi
        plot_patterns=("$@")
        ;;
    "")
        ;;
    *)
        echo "Unknown command or argument: $1" >&2
        echo >&2
        show_top_help >&2
        exit 2
        ;;
esac

module load Stages/2025 GCCcore/.13.3.0 CUDA FFmpeg/.7.0.2

cd $PLOT_DIR || { echo "Failed to change directory to $PLOT_DIR"; exit 1; }

# 定义处理单个视频的函数
process_video() {
    local simu_name_pattern="$1"
    local plot_pattern="$2"
    local start_time
    local end_time
    local duration
    local gpu_quality_params
    local cpu_quality_params
    local list_file
    local output_name
    local cuda_status

    start_time=$(date +%s) # 记录视频开始时间
    gpu_quality_params=$(get_gpu_quality_params "$plot_pattern")
    cpu_quality_params=$(get_cpu_quality_params "$plot_pattern")
    echo "======================================================"
    echo "Making video for $simu_name_pattern $plot_pattern"
    echo "GPU quality params: $gpu_quality_params"
    echo "CPU quality params: $cpu_quality_params"
    echo "Video filter: $VIDEO_FILTER"
    echo "======================================================"
    list_file="${simu_name_pattern}${plot_pattern}.txt"
    ls *"${simu_name_pattern}"*.0"${plot_pattern}" | make_ffmpeg_list -k 7 > "$list_file" #*.0 = 只考虑整的nbody时间。后续需要做成命令行参数
    output_name="${simu_name_pattern}${plot_pattern}.mp4"
    rm -f "$output_name"

    if ffmpeg_cuda_available; then
        echo "CUDA FFmpeg backend detected; using GPU encoding."
        if ! run_ffmpeg_cuda "$list_file" "$gpu_quality_params" "$output_name"; then
            echo "CUDA encoding failed; falling back to CPU encoding." >&2
            rm -f "$output_name"
            run_ffmpeg_cpu "$list_file" "$cpu_quality_params" "$output_name"
        fi
    else
        cuda_status=$?
        if [[ $cuda_status -eq 2 ]]; then
            echo "Could not detect FFmpeg CUDA support; trying CUDA first, then falling back to CPU on failure."
            if ! run_ffmpeg_cuda "$list_file" "$gpu_quality_params" "$output_name"; then
                echo "CUDA encoding failed; falling back to CPU encoding." >&2
                rm -f "$output_name"
                run_ffmpeg_cpu "$list_file" "$cpu_quality_params" "$output_name"
            fi
        else
            echo "CUDA FFmpeg backend not detected; using CPU encoding."
            run_ffmpeg_cpu "$list_file" "$cpu_quality_params" "$output_name"
        fi
    fi

    end_time=$(date +%s) # 记录视频结束时间
    duration=$((end_time - start_time)) # 计算持续时间
    echo "Time taken for $output_name: $duration seconds" # 直接打印每个视频的持续时间
}

export VIDEO_FILTER
export -f process_video get_gpu_quality_params get_cpu_quality_params make_ffmpeg_list ffmpeg_cuda_available ffmpeg_libx265_available run_ffmpeg_cpu run_ffmpeg_cuda # 导出函数以供 xargs 使用

# 生成组合并通过管道传递给 xargs 以并行执行
( # 使用子 shell 对 echo 命令进行分组
for simu_name_pattern in "${simu_name_patterns[@]}"; do
    for plot_pattern in "${plot_patterns[@]}"; do
        echo "$simu_name_pattern $plot_pattern"
    done
done
) | xargs -P "$MAX_PARALLEL_JOBS" -n 2 bash -c 'process_video "$0" "$1"'


# 记录总结束时间
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
echo "Total script execution time: $total_duration seconds"
