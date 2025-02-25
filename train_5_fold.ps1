conda activate gdcm

#python draw_froc.py

#python train3D.py --exp_name train_rc_config_5.13_from5.11+ohem_f1 --crx_valid 1 --resume --weight_path .\checkpoint\train_rc_config_5.13_from5.11+ohem_f1\backup_epoch199_tmp_save.pt --avoid_first_time_update_fp
#python train3D.py --exp_name train_rc_config_5.13_from5.11+ohem_f2 --crx_valid 2 --resume --weight_path .\checkpoint\train_rc_config_5.13_from5.11+ohem_f2\backup_epoch199_tmp_save.pt
#python train3D.py --exp_name train_rc_config_5.13.2_from5.11+ohem_f2 --crx_valid 2


python train3D.py --exp_name train_rc_config_Darknet+SSR_no_fpr_f0 --crx_valid 0 
python train3D.py --exp_name train_rc_config_Darknet+SSR_no_fpr_f1 --crx_valid 1
python train3D.py --exp_name train_rc_config_Darknet+SSR_no_fpr_f2 --crx_valid 2 
python train3D.py --exp_name train_rc_config_Darknet+SSR_no_fpr_f3 --crx_valid 3 
python train3D.py --exp_name train_rc_config_Darknet+SSR_no_fpr_f4 --crx_valid 4 


#python train3D.py --exp_name retrain_CSPDarknet_normal_fpr_f0 --crx_valid 0
#python train3D.py --exp_name retrain_CSPDarknet_normal_fpr_f1 --crx_valid 1
#python train3D.py --exp_name retrain_CSPDarknet_normal_fpr_f2 --crx_valid 2
#python train3D.py --exp_name retrain_CSPDarknet_normal_fpr_f3 --crx_valid 3
#python train3D.py --exp_name retrain_CSPDarknet_normal_fpr_f4 --crx_valid 4

#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f0 --crx_valid 0 --weight_path .\checkpoint\train_rc_config_5.8_iterative_fp_update_f0\backup_epoch85.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f1 --crx_valid 1 --weight_path .\checkpoint\train_rc_config_5.9.2_iterative_fp_update_f1\backup_epoch85.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f2 --crx_valid 2 --weight_path .\checkpoint\train_rc_config_5.9.1_iterative_fp_update_f2\backup_epoch85.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f3 --crx_valid 3 --weight_path .\checkpoint\train_rc_config_5.9.1_iterative_fp_update_f3\backup_epoch85.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f4 --crx_valid 4 --weight_path .\checkpoint\resnest_no_fp_reduction_dry_run_f4\backup_epoch187.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f1 --crx_valid 1 --weight_path .\checkpoint\train_rc_config_5.9.2_iterative_fp_update_f1\backup_epoch85.pt --resume
#python train3D.py --exp_name resnest_no_fp_reduction_dry_run_f2 --crx_valid 2 --weight_path .\checkpoint\resnest_no_fp_reduction_dry_run_f2\backup_epoch255.pt --resume

#
#python train3D.py --exp_name different_fp_data_f0 --crx_valid 0 --weight_path .\checkpoint\train_rc_config_5.6.4_resnest_shallower_f0\backup_epoch153.pt --resume
#python train3D.py --exp_name different_fp_data_f0 --crx_valid 0 --weight_path .\checkpoint\different_fp_data_f0\backup_epoch221.pt --resume
#python train3D.py --exp_name different_fp_data_f1 --crx_valid 1 --weight_path .\checkpoint\train_rc_config_5.6.5_resnest_shallower_f1\backup_epoch153.pt --resume
#python train3D.py --exp_name different_fp_data_f2 --crx_valid 2 --weight_path .\checkpoint\train_rc_config_5.6_resnest+sgd_shallower_f2\backup_epoch153.pt --resume
#python train3D.py --exp_name different_fp_data_f3 --crx_valid 3 --weight_path .\checkpoint\train_rc_config_5.6_resnest+sgd_shallower_f3\backup_epoch153.pt --resume
#python train3D.py --exp_name different_fp_data_f4 --crx_valid 4 --weight_path .\checkpoint\train_rc_config_5.6.2_resnest_shallower_f4\backup_epoch153.pt --resume


#python train3D.py --exp_name train_fake_1.25mm_from_2.5mm_config_1_f0 --crx_valid 0
#python train3D.py --exp_name train_fake_1.25mm_from_2.5mm_config_1.2_f2 --crx_valid 2
#python train3D.py --exp_name train_fake_1.25mm_from_2.5mm_config_1.2_f1 --crx_valid 1
#python train3D.py --exp_name train_fake_1.25mm_from_2.5mm_config_1_f3 --crx_valid 3
#python train3D.py --exp_name train_fake_1.25mm_from_2.5mm_config_1_f4 --crx_valid 4


#python train3D.py --exp_name train_fake_1.25mm_config_1.4_f4 --crx_valid 4
#python train3D.py --exp_name train_fake_1.25mm_config_1.5_f4 --crx_valid 4

#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f1  --crx_valid 1 
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f2  --crx_valid 2 
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f3  --crx_valid 3 --resume --weight_path .\checkpoint\train_5mm_max_no_fp_reduction_dry_run_f3\backup_epoch85.pt
#python train3D.py --exp_name train_5mm_max_no_fp_reduction_dry_run_f4  --crx_valid 4
