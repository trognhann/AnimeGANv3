---
name: AnimeGANv3 Management
description: Instructions and commands for training, testing, and managing the AnimeGANv3 workflows in this project.
---

# AnimeGANv3 Project Skill

This skill provides guidelines and commands for interacting with the AnimeGANv3 workspace. As an agent, follow these instructions when the user wants to train new models, test existing checkpoints, or configure datasets.

## 1. Project Structure
- `train.py`: Main script for training the AnimeGANv3 model.
- `test.py`: Main script for running inference (testing) with a trained model.
- `AnimeGANv3_hayao.py`, `AnimeGANv3_shinkai.py`: Model architecture definitions and training loops for specific styles.
- `checkpoint/`: Directory containing saved model weights.
- `inputs/imgs/`: Default directory for raw input images used in testing.
- `style_results/`: Default directory where stylized output images are saved.
- `dataset/`: Directory containing the training dataset (real images and style images).
- `log/`: Directory storing training logs (e.g., `hayao-notebook.log`).

## 2. Training the Model (`train.py`)

When the user wants to train the model, use `train.py`.
### Key Arguments
- `--style_dataset`: The name of the style dataset (e.g., `Hayao`, `Shinkai`). Default: `Hayao`.
- `--epoch`: Total number of training epochs. Default: `60`.
- `--init_G_epoch`: Epochs for generator initialization (pre-training without discriminator). Default: `5`.
- `--batch_size`: Batch size for training. Default: `8`.
- `--save_freq`: Frequency of saving checkpoints (in epochs). Default: `4`.
- `--checkpoint_dir`: Where to save checkpoints. Default: `checkpoint`.
- `--load_or_resume`: `load` for fine-tuning, `resume` to resume training. Default: `load`.

### Example Command:
```bash
python train.py --style_dataset Hayao --epoch 60 --batch_size 8 --checkpoint_dir checkpoint/hayao_run
```

## 3. Testing/Inference (`test.py`)

When the user wants to stylize images using a trained model, use `test.py`.
### Key Arguments
- `--checkpoint_dir`: Path to the directory containing the saved model checkpoint. Default: `checkpoint/checkpoint_hayao`.
- `--test_dir`: Directory containing the input images to be stylized. Default: `inputs/imgs`.
- `--save_dir`: Directory where the resulting stylized images will be saved. Default: `style_results/`.

### Example Command:
```bash
python test.py --checkpoint_dir checkpoint/checkpoint_hayao --test_dir inputs/imgs --save_dir style_results/
```

### Note on Outputs:
The test script outputs multiple versions of the stylized image for each input:
- `a_*`: Original input
- `b_*`: Stylized image (with guided filter)
- `c_*`: Raw generator output
- `d_*`: Main Generator support output (if applicable)

## 4. Troubleshooting and Logs
- If training behavior is unexpected (e.g., "identity mapping" where the generator outputs the input without strong stylization), check the loss logs.
- The `log/` directory contains standard outputs. A high `G_loss` (> 5-8) combined with a very low `D_loss` (< 0.2) indicates the Discriminator is overpowering the Generator.
- To analyze logs, parsing scripts can be used to plot `Pre_train_G_loss`, `G_loss`, `D_loss`, and `G_support_loss` over time.

## 5. Resources and Compliance
- **Architecture Guidelines**: Any modifications, reviews, or debugging of the model architecture must strictly adhere to the guidelines and architecture described in the original manuscript: `resource/AnimeGANv3_manuscript.pdf`.
