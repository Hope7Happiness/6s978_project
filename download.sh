# rsync -av mitgpu:/nobackup/users/zhh24/dev/6s978_proj/checkpoints/ep40_sanity.pth ./checkpoints/ep40_vae.pth
rsync -av ./remote_samples/samples/ep40_sanity_data.png jzc:/home/zhh/zhh/6s978_project/samples
rsync -av ./remote_samples/samples/ep40_sanity_recon.png jzc:/home/zhh/zhh/6s978_project/samples
rsync -av ./remote_samples/samples/ep40_sanity_gen.png jzc:/home/zhh/zhh/6s978_project/samples
# rsync -av ./checkpoints/ep40_vae.pth jzc:/home/zhh/zhh/6s978_project 