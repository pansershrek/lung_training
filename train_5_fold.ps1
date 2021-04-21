conda activate gdcm

#
python train3D.py --exp_name train_rc_config_5.8.1_SCResNeSt_SEConv_f0 --crx_valid 0
python train3D.py --exp_name train_rc_config_5.8.1_SCResNeSt_SEConv_f1 --crx_valid 1
python train3D.py --exp_name train_rc_config_5.8.1_SCResNeSt_SEConv_f2 --crx_valid 2
python train3D.py --exp_name train_rc_config_5.8.1_SCResNeSt_SEConv_f3 --crx_valid 3
python train3D.py --exp_name train_rc_config_5.8.1_SCResNeSt_SEConv_f4 --crx_valid 4


#python train3D.py --exp_name train_fake_1.25mm_config_1.4_f4 --crx_valid 4
#python train3D.py --exp_name train_fake_1.25mm_config_1.5_f4 --crx_valid 4

#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f1  --crx_valid 1 
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f2  --crx_valid 2 
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f3  --crx_valid 3 --resume --weight_path .\checkpoint\train_5mm_max_no_fp_reduction_dry_run_f3\backup_epoch85.pt
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f4  --crx_valid 4

#python train3D.py --exp_name train_rc_config_5.6.4_resnest_shallower_f0 --crx_valid 0 --resume --weight_path .\checkpoint\train_rc_config_5.6.4_resnest_shallower_f0\backup_epoch255.pt
#python train3D.py --exp_name train_rc_config_5.6.4_resnest_shallower_f1 --crx_valid 1
#python train3D.py --exp_name train_rc_config_5.6.5_resnest_shallower_f0 --crx_valid 0 
#python train3D.py --exp_name train_rc_config_5.6.5_resnest_shallower_f1 --crx_valid 1

#python train3D.py --exp_name train_rc_config_4_fp_pool_f0 --crx_valid 0  ## pos_weight=0.5 (low fp but low tp either)
#python train3D.py --exp_name train_rc_config_4_fp_pool_f1 --crx_valid 1 --resume --weight_path checkpoint/train_rc_config_4_fp_pool_f1/backup_epoch119.pt  ## change pos_weight to dynamic (due to )
#python train3D.py --exp_name train_rc_config_4_fp_pool_f2 --crx_valid 2
#python train3D.py --exp_name train_rc_config_4_fp_pool_f3 --crx_valid 3
#python train3D.py --exp_name train_rc_config_4_fp_pool_f4 --crx_valid 4