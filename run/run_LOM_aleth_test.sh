# buu
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_buu.gin --logbase ./logs --ginb run.run_train=False --ginb run.run_render=True
#chair
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_chair.gin --logbase ./logs --ginb run.run_train=False --ginb run.run_render=True 
# sofa
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_sofa.gin --logbase ./logs --ginb run.run_train=False --ginb run.run_render=True 
# bike
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_bike.gin --logbase ./logs --ginb run.run_train=False --ginb run.run_render=True 
# shrub
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_shrub.gin --logbase ./logs --ginb run.run_train=False --ginb run.run_render=True 
