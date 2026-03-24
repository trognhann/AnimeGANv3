import re
import matplotlib.pyplot as plt
import os

log_file = '/Users/trognhann/Desktop/AnimeGANv3/log/hayao-notebook.log'
out_dir = '/Users/trognhann/.gemini/antigravity/brain/73a1d14e-034a-49b4-bdb8-dcb12a0e962b'

pre_train_loss = []
d_loss = []
g_loss = []
g_support_loss = []
g_main_loss = []

pre_pattern = re.compile(r'Pre_train_G_loss:\s*([0-9.]+)')
gan_pattern = re.compile(r'D_loss:\s*([0-9.]+)\s+~\s+G_loss:\s+([0-9.]+)\s+\|\|\s+G_support_loss:\s+([0-9.]+).*?G_main_loss:\s+([0-9.]+)')

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        m_pre = pre_pattern.search(line)
        if m_pre:
            pre_train_loss.append(float(m_pre.group(1)))
        else:
            m_gan = gan_pattern.search(line)
            if m_gan:
                d_loss.append(float(m_gan.group(1)))
                g_loss.append(float(m_gan.group(2)))
                g_support_loss.append(float(m_gan.group(3)))
                g_main_loss.append(float(m_gan.group(4)))

# Plot Pre-train G Loss
plt.figure(figsize=(10, 5))
plt.plot(pre_train_loss, label='Pre-train G Loss', color='blue', alpha=0.7)
plt.title('Pre-train Generator Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'pre_train_loss.png'))

# Plot GAN Losses (D_loss and G_loss)
plt.figure(figsize=(10, 5))
plt.plot(d_loss, label='D Loss', color='red', alpha=0.7)
plt.plot(g_loss, label='G Loss', color='green', alpha=0.7)
plt.title('GAN D Loss and G Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gan_loss.png'))

# Plot G Sub-Losses
plt.figure(figsize=(10, 5))
plt.plot(g_support_loss, label='G Support Loss', color='purple', alpha=0.7)
plt.plot(g_main_loss, label='G Main Loss', color='orange', alpha=0.7)
plt.title('Generator Support vs Main Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gan_sub_loss.png'))

print(f"Parsed {len(pre_train_loss)} pre-train steps and {len(g_loss)} GAN steps.")
print("Plots saved successfully.")
