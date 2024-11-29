rsync -av . jzc:/home/zhh/zhh/6s978_project --exclude-from=".gitignore" --exclude="*.ipynb" --exclude=".git"
# rsync -av . mitgpu:/nobackup/users/zhh24/dev/6s978_proj --exclude-from=".gitignore" --exclude="*.ipynb" --exclude=".git"
# rsync -av checkpoints/ep20_pixelcnn_torch1.2.pth mitgpu:/nobackup/users/zhh24/dev/6s978_proj/checkpoints