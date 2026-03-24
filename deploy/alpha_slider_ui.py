"""
deploy/alpha_slider_ui.py
=========================
Interactive demo for Controllable-LADE (C-LADE) AnimeGANv3.

Usage:
    python deploy/alpha_slider_ui.py --checkpoint_dir checkpoint/checkpoint_hayao

The Gradio UI lets you:
  - Upload any portrait photo
  - Drag the α slider from 0.0 (original photo) to 1.0 (full anime style)
  - See the stylized result update in real time

Requirements:
    pip install gradio
"""

import argparse
import os
import sys
import numpy as np
import cv2

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from net import generator
from tools.GuidedFilter import guided_filter


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigm_out_scale(x):
    return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)

def tanh_out_scale(x):
    return tf.clip_by_value((x - 0.5) * 2.0, -1.0, 1.0)

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Resize to multiple-of-8, normalise to [-1, 1]."""
    h, w = img_rgb.shape[:2]
    def to_8s(x):
        return 256 if x < 256 else x - x % 8
    img = cv2.resize(img_rgb, (to_8s(w), to_8s(h)))
    return img.astype(np.float32) / 127.5 - 1.0

def postprocess(tensor: np.ndarray, original_hw) -> np.ndarray:
    """Convert [-1,1] tensor → uint8 RGB, resized to original dims."""
    img = (tensor.squeeze() + 1.0) / 2.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (original_hw[1], original_hw[0]))
    return img


# ── TF Graph (built once) ─────────────────────────────────────────────────────

class StyleEngine:
    def __init__(self, checkpoint_dir: str):
        self.graph = tf.Graph()
        self.sess  = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.input_ph = tf.placeholder(tf.float32, [1, None, None, 3], name='input')
            self.alpha_ph = tf.placeholder(tf.float32, [], name='style_alpha')

            with tf.variable_scope("generator", reuse=False):
                # Build with alpha=1 to initialise all variables
                _, _ = generator.G_net(self.input_ph, False, alpha=self.alpha_ph)

            # Variables for restore
            variables = tf.global_variables()
            gen_vars  = [v for v in variables
                         if v.name.startswith('generator') and 'Adam' not in v.name]
            saver = tf.train.Saver(gen_vars)

            with self.graph.as_default():
                with tf.variable_scope("generator", reuse=True):
                    test_s0, test_m = generator.G_net(self.input_ph, False, alpha=self.alpha_ph)
                    self.output = tanh_out_scale(
                        guided_filter(sigm_out_scale(test_s0),
                                      sigm_out_scale(test_s0), 2, 0.01))

            self.sess.run(tf.global_variables_initializer())

        # Restore checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(f" [*] Loaded checkpoint: {ckpt.model_checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in '{checkpoint_dir}'. "
                "Train the model first or specify the correct --checkpoint_dir.")

    def stylize(self, img_rgb: np.ndarray, alpha: float) -> np.ndarray:
        """img_rgb: H×W×3 uint8 →  H×W×3 uint8 stylized."""
        original_hw = img_rgb.shape[:2]
        inp = preprocess(img_rgb)[np.newaxis]          # [1, H', W', 3]
        out = self.sess.run(self.output,
                            feed_dict={self.input_ph: inp,
                                       self.alpha_ph: float(np.clip(alpha, 0.0, 1.0))})
        return postprocess(out, original_hw)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(engine: StyleEngine):
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Please install Gradio:  pip install gradio")

    def process(image, alpha):
        if image is None:
            return None
        stylized = engine.stylize(image, alpha)
        return stylized

    with gr.Blocks(title="AnimeGANv3 – C-LADE Style Controller",
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "## 🎨 AnimeGANv3 Controllable Style Transfer\n"
            "Upload a portrait photo and drag the **Style Strength (α)** slider\n"
            "to blend between the original photo (α=0) and full anime style (α=1)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Input Photo", type="numpy")
                alpha_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=1.0, label="Style Strength (α)",
                    info="0 = original photo  •  1 = full anime"
                )
                btn = gr.Button("Stylize ▶", variant="primary")

            with gr.Column(scale=1):
                img_output = gr.Image(label="Stylized Output", type="numpy")

        # Trigger on button click
        btn.click(fn=process,
                  inputs=[img_input, alpha_slider],
                  outputs=img_output)

        # Also trigger live on slider change (when image is uploaded)
        alpha_slider.change(fn=process,
                            inputs=[img_input, alpha_slider],
                            outputs=img_output)

        gr.Markdown(
            "---\n"
            "**C-LADE Formula:** `output = (1 − α) · content + α · style_LADE`  \n"
            "**Adversarial loss** is active only when α > 0.5 to preserve identity at low alpha."
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANv3 C-LADE Slider UI")
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/checkpoint_hayao',
                        help='Path to trained checkpoint directory')
    parser.add_argument('--port', type=int, default=7860,
                        help='Gradio server port')
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio share link')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f" [*] Loading model from: {args.checkpoint_dir}")
    engine = StyleEngine(args.checkpoint_dir)
    demo   = build_ui(engine)
    print(f" [*] Starting Gradio UI on port {args.port} ...")
    demo.launch(server_port=args.port, share=args.share)
