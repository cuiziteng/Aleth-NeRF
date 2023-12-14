# buu
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf_exp/aleth_nerf_buu.gin --logbase ./logs_exp --con 2    
# chair
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf_exp/aleth_nerf_chair.gin --logbase ./logs_exp --con 2
# sofa
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf_exp/aleth_nerf_sofa.gin --logbase ./logs_exp --con 2
# bike
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf_exp/aleth_nerf_bike.gin --logbase ./logs_exp --con 2 
# shrub
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf_exp/aleth_nerf_shrub.gin --logbase ./logs_exp --con 2 
                          