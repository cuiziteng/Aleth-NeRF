# buu
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_buu.gin --logbase ./logs 
#chair
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_chair.gin --logbase ./logs 
# sofa
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_sofa.gin --logbase ./logs 
# bike
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_bike.gin --logbase ./logs 
# shrub
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_shrub.gin --logbase ./logs 
