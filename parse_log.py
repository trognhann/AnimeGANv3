import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

log_file = r"d:\Coding\GAN\AnimeGANv3\log\gantf2.log"
output_dir = r"d:\Coding\GAN\AnimeGANv3\log_analysis"
os.makedirs(output_dir, exist_ok=True)

pre_train_losses = []
d_losses = []
g_losses = []
g_support_losses = []
d_support_losses = []
g_main_losses = []
d_main_losses = []
sty_losses = []
color_losses = []
con_losses = []
s22_losses = []
s33_losses = []
s44_losses = []

# Regex patterns
# E.g. 130: 102.9s 130 Epoch:   0, Step:     0 /   180, time: 12.986s, ETA: 2350.49s, Pre_train_G_loss: 0.932500
pretrain_pattern = re.compile(r"Pre_train_G_loss:\s+([0-9.]+)")

# E.g. Epoch:   5, Step:     0 /  180, time: 26.015s, ETA: 4708.79s, D_loss:0.566 ~ G_loss: 28.362 || G_support_loss: 5.869624, g_s_loss: 0.813537, con_loss: 0.093651, rs_loss: 0.836310, sty_loss: 3.136441, s22: 0.085268, s33: 0.652115, s44: 2.399058, color_loss: 0.989683, tv_loss: 0.000002 ~ D_support_loss: 0.466153 || G_main_loss: 22.491949, g_m_loss: 0.020011, p0_loss: 21.945515, p4_loss: 0.526423, tv_loss_m: 0.000001 ~ D_main_loss: 0.100049
gan_pattern = re.compile(
    r"D_loss:([0-9.]+)\s*~\s*G_loss:\s*([0-9.]+).*?"
    r"G_support_loss:\s*([0-9.]+).*?"
    r"con_loss:\s*([0-9.]+).*?"
    r"sty_loss:\s*([0-9.]+),\s*s22:\s*([0-9.]+),\s*s33:\s*([0-9.]+),\s*s44:\s*([0-9.]+),\s*color_loss:\s*([0-9.]+).*?"
    r"D_support_loss:\s*([0-9.]+).*?"
    r"G_main_loss:\s*([0-9.]+).*?"
    r"D_main_loss:\s*([0-9.]+)"
)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # Check Pretrain
        m_pre = pretrain_pattern.search(line)
        if m_pre:
            pre_train_losses.append(float(m_pre.group(1)))
            continue
            
        # Check GAN
        m_gan = gan_pattern.search(line)
        if m_gan:
            d_losses.append(float(m_gan.group(1)))
            g_losses.append(float(m_gan.group(2)))
            g_support_losses.append(float(m_gan.group(3)))
            con_losses.append(float(m_gan.group(4)))
            sty_losses.append(float(m_gan.group(5)))
            s22_losses.append(float(m_gan.group(6)))
            s33_losses.append(float(m_gan.group(7)))
            s44_losses.append(float(m_gan.group(8)))
            color_losses.append(float(m_gan.group(9)))
            d_support_losses.append(float(m_gan.group(10)))
            g_main_losses.append(float(m_gan.group(11)))
            d_main_losses.append(float(m_gan.group(12)))

print(f"Parsed {len(pre_train_losses)} pretrain steps.")
print(f"Parsed {len(d_losses)} GAN steps.")

# Plot 1: Pretrain Loss
plt.figure(figsize=(10, 5))
plt.plot(pre_train_losses, label="Pre_train_G_loss", color='blue', alpha=0.7)
plt.title("Pre-training Generator Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pretrain_g_loss.png"))
plt.close()

# Plot 2: Total D vs Total G Loss
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label="D_loss", color='red', alpha=0.7)
plt.plot(g_losses, label="G_loss", color='blue', alpha=0.7)
plt.title("Total Generator vs Discriminator Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_gan_losses.png"))
plt.close()

# Plot 3: Specific Support Losses (Style, Content, Color)
plt.figure(figsize=(12, 6))
plt.plot(sty_losses, label="Style Loss", color='purple', alpha=0.7)
plt.plot(con_losses, label="Content Loss", color='orange', alpha=0.7)
plt.plot(color_losses, label="Color Loss", color='green', alpha=0.7)
plt.title("Generator Components Loss (Style, Content, Color)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "g_components_loss.png"))
plt.close()

# Plot 4: Style Sub-Losses (s22, s33, s44)
plt.figure(figsize=(12, 6))
plt.plot(s22_losses, label="s22 (VGG/layer2)", alpha=0.7)
plt.plot(s33_losses, label="s33 (VGG/layer3)", alpha=0.7)
plt.plot(s44_losses, label="s44 (VGG/layer4)", alpha=0.7)
plt.title("Style Sub-Losses (VGG Feature Layers)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "style_sub_losses.png"))
plt.close()

# Plot 5: Main vs Support Network Losses
plt.figure(figsize=(12, 6))
plt.plot(g_main_losses, label="G Main Loss", alpha=0.7)
plt.plot(g_support_losses, label="G Support Loss", alpha=0.7)
plt.plot(d_main_losses, label="D Main Loss", alpha=0.7)
plt.plot(d_support_losses, label="D Support Loss", alpha=0.7)
plt.title("Main vs Support Network Losses")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.yscale('log') # Use log scale because main loss is typically larger
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "main_vs_support_losses.png"))
plt.close()

print(f"Plots saved successfully to {output_dir}")
