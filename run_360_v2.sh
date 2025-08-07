set -xe

# module load cuda

# source ~/.bashrc
# conda activate cosmos-predict1

DATASET_NAME="360_v2"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
BUFFER_LIST="basecolor depth metallic normal roughness"

# mkdir -p data/examples/${DATASET_NAME}_examples
# for SCENE in $SCENE_LIST; do
#     cp ../gsplat/examples/results/benchmark_mcmc_4M/$SCENE/videos/traj_29999.mp4 data/examples/${DATASET_NAME}_examples/$SCENE.mp4
# done

# python scripts/dataproc_extract_frames_from_video.py \
#     --input_folder data/examples/${DATASET_NAME}_examples/ \
#     --output_folder data/examples/${DATASET_NAME}_frames_examples/ \
#     --frame_rate 24 \
#     --resize 1280x704 # --max_frames=57
    

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=data/examples/${DATASET_NAME}_frames_examples/ \
    --chunk_mode first \
    --num_video_frames 512 \
    --group_mode folder \
    --video_save_folder=data/${DATASET_NAME}_examples_results/video_delighting/ 

    # --inference_passes metallic \

    

# for SCENE in $SCENE_LIST; do
#     RESULTS_DIR="./data/${DATASET_NAME}_examples_results"
#     mkdir -p $RESULTS_DIR/video_concat

#     # Get width and height of the first video
#     FIRST_VIDEO="./data/examples/${DATASET_NAME}_examples/$SCENE.mp4"
#     WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$FIRST_VIDEO")
#     HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$FIRST_VIDEO")

#     ffmpeg -y \
#         -i $FIRST_VIDEO \
#         -i "$RESULTS_DIR/video_delighting/$SCENE.0000.depth.mp4" \
#         -i "$RESULTS_DIR/video_delighting/$SCENE.0000.normal.mp4" \
#         -i "$RESULTS_DIR/video_delighting/$SCENE.0000.basecolor.mp4" \
#         -i "$RESULTS_DIR/video_delighting/$SCENE.0000.metallic.mp4" \
#         -i "$RESULTS_DIR/video_delighting/$SCENE.0000.roughness.mp4" \
#         -filter_complex "\
#             [0:v]scale=${WIDTH}:${HEIGHT}[v0]; \
#             [1:v]scale=${WIDTH}:${HEIGHT}[v1]; \
#             [2:v]scale=${WIDTH}:${HEIGHT}[v2]; \
#             [3:v]scale=${WIDTH}:${HEIGHT}[v3]; \
#             [4:v]scale=${WIDTH}:${HEIGHT}[v4]; \
#             [5:v]scale=${WIDTH}:${HEIGHT}[v5]; \
#             [v0][v1][v2]hstack=3[top]; \
#             [v3][v4][v5]hstack=3[bottom]; \
#             [top][bottom]vstack=2[v]" \
#         -map "[v]" \
#         -c:v libx264 \
#         -crf 16 \
#         -preset slow \
#         -profile:v high \
#         -pix_fmt yuv420p \
#         "$RESULTS_DIR/video_concat/$SCENE.mp4"
# done







# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#     --checkpoint_dir checkpoints \
#     --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#     --dataset_path=asset/${DATASET_NAME}_examples_results/video_delighting/gbuffer_frames \
#     --num_video_frames 57 \
#     --envlight_ind 0 1 2 3 --use_custom_envmap=True \
#     --video_save_folder=asset/${DATASET_NAME}_examples_results/video_relighting/

    # ffmpeg -y -i $FIRST_VIDEO -filter:v "setpts=PTS*1.25" -r 24 "$RESULTS_DIR/video_delighting/$SCENE.color.mp4"
        # -i "$RESULTS_DIR/video_delighting/$SCENE.color.mp4" \




    # # Concat buffer videos
    # for BUFFER_NAME in $BUFFER_LIST; do
    #     # Get sorted list of matching video files
    #     files=($(find "$RESULTS_DIR/video_delighting" -name "$SCENE.*.$BUFFER_NAME.mp4" | sort))
    #     N=${#files[@]}

    #     # Build ffmpeg input arguments
    #     inputs=()
    #     filter=""
    #     for i in "${!files[@]}"; do
    #         inputs+=("-i" "${files[$i]}")
    #         filter+="[$i:v:0]"
    #     done

    #     filter+="concat=n=$N:v=1:a=0[outv]"

    #     # Run ffmpeg to concatenate with re-encoding
    #     ffmpeg -y \
    #         "${inputs[@]}" \
    #         -filter_complex "$filter" \
    #         -map "[outv]" \
    #         $RESULTS_DIR/video_concat/$SCENE.$BUFFER_NAME.mp4
    # done