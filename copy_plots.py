import shutil
import glob
import os

src_dir = r"d:\Coding\GAN\AnimeGANv3\log_analysis"
dst_dir = r"C:\Users\ASUS ZENBOOK\.gemini\antigravity\brain\3f497180-1d6a-4dc4-a5cc-ce7eacaa2a30"

print(f"Copying from {src_dir} to {dst_dir}")
for file in glob.glob(os.path.join(src_dir, "*.png")):
    shutil.copy(file, dst_dir)
    print(f"Copied {file}")
