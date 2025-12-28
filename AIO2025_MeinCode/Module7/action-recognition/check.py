import os
import sys
import torch
# import torchcodec

# Windows 11 specific: Explicitly add the FFmpeg bin path to the DLL search path
# Change this to your actual FFmpeg bin folder
# ffmpeg_path = r"C:\ffmpeg\bin"

# if os.path.exists(ffmpeg_path):
#     os.add_dll_directory(ffmpeg_path)
# else:
#     print(f"Warning: FFmpeg path {ffmpeg_path} not found.")

import torch
print(torch.version.cuda)
print(f"Successfully loaded TorchCodec with CUDA: {torch.cuda.is_available()}")