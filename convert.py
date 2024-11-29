import torch
f = torch.load('checkpoints/ep20_pixelcnn.pth', map_location='cpu')
torch.save(f, 'checkpoints/ep20_pixelcnn_torch1.2.pth', _use_new_zipfile_serialization=False)
