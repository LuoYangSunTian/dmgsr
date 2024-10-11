# noise-free degradations with isotropic Gaussian blurs
##  GPU 0,1
#python HATBmain.py --dir_data='./Train/' \
#               --model='DMGSR_param1' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DMGSR_param1' \
#               --start_epoch=515 \
#               --resume=516

python main.py --dir_data='./Train' \
               --model='DMGSR' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0 \
               --save='DMGSR' \
               --resume=0 \
               --start_epoch=0

#python HATBmain.py --dir_data='./Train/' \
#               --model='DGFT1' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DGFT1' \
#               --start_epoch=508 \
#               --resume=509


#python HATBmain.py --dir_data='./Train/' \
#               --model='HATBGAN' \
#               --scale='3' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=3.0 \
#               --save='HATBGAN' \
#               --start_epoch=600 \
#               --resume=600

#python HATBmain.py --dir_data='./Train/' \
#               --model='DMGSR_loss_weight9' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DMGSR_loss_weight9' \
#               --start_epoch=631 \
#               --resume=632


#python HATBmain.py --dir_data='./Train/' \
#               --model='DMGSR_loss_weight1' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DMGSR_loss_weight1' \
#               --start_epoch=630 \
#               --resume=630
##
#
#python HATBmain.py --dir_data='./Train/' \
#               --model='DMGSR_loss_weight3' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DMGSR_loss_weight3' \
#               --start_epoch=630 \
#               --resume=574



#  GPU 2,3
#python HATBmain.py --dir_data='./Train/' \
#               --model='DMGSR_DGFT_First' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='DMGSR_DGFT_First' \
#               --start_epoch=598 \
#               --resume=599

#  GPU 4,5


#python main.py --dir_data='./Train/' \
#               --model='simsft_one_only_pixel' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='simsft_one_only_pixel' \
#               --start_epoch=0


#python main.py --dir_data='./Train/' \
#               --model='simsft_one_plus' \
#               --scale='4' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0 \
#               --save='simsft_one_plus' \
#               --start_epoch=199 \
#               --resume=200

#python mainRCA.py --dir_data='./Train/' \
#               --model='MSCN32' \
#               --scale='2' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig_min=0.2 \
#               --sig_max=2.0 \
#               --save='MSCN32'
#python main.py --dir_data='./Train/' \
#               --model='Ours' \
#               --scale='2' \
#               --blur_type='aniso_gaussian' \
#               --noise=25.0 \
#               --lambda_min=0.2 \
#               --lambda_max=4.0 \
#               --save='Ours' \
#               --start_epoch=99

#python main.py --dir_data='./Train/'\
#               --model='HAT' \
#               --scale='2' \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --resume=196 \
#               --sig_min=0.2 \
#               --sig_max=2.0 \
#               --save='HAT' \
#               --start_epoch=99