CUDA_VISIBLE_DEVICES=3 python scripts/MSDFont_eval.py \
--outdir results/infer_result \
--path_genchar FontData/chn/eval_unseen_chars.json \
--path_refchar FontData/chn/ref_chars.json \
--path_ttf FontData/chn/ttfs/infer_unseen_font \
--source_path FontData/chn/source.ttf \
--path_config_rec configs/MSDFont/MSDFont_Eval_rec_model_predx0_miniUnet.yaml \
--path_rec_model StableDiffusion/logs/stage2/checkpoints/last.ckpt \
--path_config_trans configs/MSDFont/MSDFont_Eval_trans_model_predx0_miniUnet.yaml \
--path_trans_model StableDiffusion/logs/stage1_1/checkpoints/last.ckpt

