# Settings shared by the pipeline scripts. Source from the repository root.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} MUJOCO_GL=${MUJOCO_GL:-disable}
WRITING_TASKS=L1_trace_line,L1_trace_circle,L1_trace_shape,L2_write_digits,L2_write_letters,L2_write_word,L2_write_chinese
ZERO_SHOT_TASKS=E2_write_word_unseen,E2_write_idiom,E2_write_chinese_unseen   # held-out content, evaluation only
GRASP=checkpoints/grasp.pt                    # learned pen grasp (trajectory-input format)
# environment settings used for every evaluation in the paper
EVAL_ENV='{"grip_assist":0.0,"start_height":[0.002,0.006],"start_wrist":true,"constraint":"idle_lock","reward":{"__preset__":"v15_goal","adv_skip":0.25},"end_mode":"policy","cursor_obs":false,"traj_obs":16,"adv_mode":"tip"}'
