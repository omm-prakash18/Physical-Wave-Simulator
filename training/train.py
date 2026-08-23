"""
Main training loop for the Latent Video Prediction model.

Features:
  - Mixed precision (fp16) training with gradient scaling
  - Gradient accumulation for effective larger batch sizes
  - Scheduled sampling (teacher forcing decay)
  - Cosine warmup LR schedule
  - TensorBoard logging of all loss components
  - Checkpoint saving (best + periodic)
  - Validation with GIF generation every eval interval
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import create_dataloaders
from model.full_model import LatentVideoPredictor
from model.losses import CombinedLoss
from training.scheduler import CosineWarmupScheduler, TeacherForcingScheduler
from training.evaluate import evaluate_model


def train(config: Config):
    """Main training function."""
    device = config.device
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.log_dir, exist_ok=True)

    # ─── Data ───────────────────────────────────────────
    print("Loading data...")
    loaders = create_dataloaders(
        data_dir=config.data.data_dir,
        t_in=config.model.t_in,
        t_out=config.model.t_out,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    print(f"  Train: {len(train_loader.dataset)} sequences, {len(train_loader)} batches")
    if val_loader:
        print(f"  Val:   {len(val_loader.dataset)} sequences, {len(val_loader)} batches")

    # ─── Model ──────────────────────────────────────────
    print("Building model...")
    model = LatentVideoPredictor(
        in_channels=config.data.channels,
        d_model=config.model.d_model,
        encoder_channels=config.model.encoder_channels,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        t_in=config.model.t_in,
        t_out=config.model.t_out,
    ).to(device)

    param_counts = model.count_parameters()
    print(f"  Encoder:     {param_counts['encoder']:>10,} params")
    print(f"  Transformer: {param_counts['transformer']:>10,} params")
    print(f"  Decoder:     {param_counts['decoder']:>10,} params")
    print(f"  Total:       {param_counts['total']:>10,} params")

    # ─── Loss ───────────────────────────────────────────
    criterion = CombinedLoss(
        encoder=model.encoder,
        lambda_recon=config.train.lambda_recon,
        lambda_latent=config.train.lambda_latent,
        lambda_perceptual=config.train.lambda_perceptual,
        lambda_temporal=config.train.lambda_temporal,
    ).to(device)

    # ─── Optimizer ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=config.train.betas,
    )

    total_steps = len(train_loader) * config.train.epochs
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.train.warmup_steps,
        total_steps=total_steps,
    )

    tf_scheduler = TeacherForcingScheduler(total_epochs=config.train.epochs)

    # ─── Mixed precision ────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=(config.train.use_amp and device == "cuda"))
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32

    # ─── Logging ────────────────────────────────────────
    run_name = f"wave_{config.model.d_model}d_{config.model.n_layers}L_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(config.train.log_dir, run_name))

    # Save config
    config_path = os.path.join(config.train.log_dir, run_name, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model": {
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "n_heads": config.model.n_heads,
                "d_ff": config.model.d_ff,
                "t_in": config.model.t_in,
                "t_out": config.model.t_out,
            },
            "train": {
                "lr": config.train.lr,
                "batch_size": config.train.batch_size,
                "epochs": config.train.epochs,
            },
            "params": param_counts,
        }, f, indent=2)

    # ─── Training loop ──────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    print(f"\nStarting training for {config.train.epochs} epochs...")
    print(f"  Run: {run_name}")
    print(f"  Effective batch size: {config.train.batch_size * config.train.grad_accumulation_steps}")
    print()

    for epoch in range(config.train.epochs):
        model.train()
        epoch_losses = {
            "loss/total": 0.0,
            "loss/recon": 0.0,
            "loss/latent": 0.0,
            "loss/perceptual": 0.0,
            "loss/temporal": 0.0,
        }
        num_batches = 0
        epoch_start = time.time()

        # Get teacher forcing ratio for this epoch
        tf_ratio = tf_scheduler.get_ratio(epoch)

        # Dynamic rollout curriculum based on epochs
        if epoch < 5:
            curr_steps = 3
        elif epoch < 10:
            curr_steps = 5
        elif epoch < 15:
            curr_steps = 7
        else:
            curr_steps = 10

        for batch_idx, batch in enumerate(train_loader):
            context = batch["context"].to(device)
            target = batch["target"].to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=(config.train.use_amp and device == "cuda"), dtype=autocast_dtype):
                output = model(
                    context_frames=context,
                    target_frames=target,
                    teacher_forcing_ratio=tf_ratio,
                )

                # Curriculum slicing on rollout horizon
                loss, components = criterion(
                    predicted_frames=output["predicted_frames"][:, :curr_steps],
                    target_frames=target[:, :curr_steps],
                    predicted_latents=output["predicted_latents"][:, :curr_steps],
                    target_latents=output["target_latents"][:, :curr_steps],
                )

                # Scale for gradient accumulation
                loss = loss / config.train.grad_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % config.train.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)

                # Calculate layer group gradient norms independently
                enc_grad = sum(p.grad.norm().item() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
                trans_grad = sum(p.grad.norm().item() ** 2 for p in model.transformer.parameters() if p.grad is not None) ** 0.5
                dec_grad = sum(p.grad.norm().item() ** 2 for p in model.decoder.parameters() if p.grad is not None) ** 0.5

                if global_step % config.train.log_interval == 0:
                    writer.add_scalar("gradients/encoder", enc_grad, global_step)
                    writer.add_scalar("gradients/transformer", trans_grad, global_step)
                    writer.add_scalar("gradients/decoder", dec_grad, global_step)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

            # Accumulate epoch losses
            for k, v in components.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            num_batches += 1

            # Per-step logging
            if global_step % config.train.log_interval == 0 and global_step > 0:
                for k, v in components.items():
                    writer.add_scalar(k, v, global_step)
                writer.add_scalar("schedule/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("schedule/teacher_forcing", tf_ratio, global_step)
                writer.add_scalar("schedule/curriculum_steps", curr_steps, global_step)

        # ─── Epoch summary ──────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

        print(
            f"Epoch {epoch + 1:3d}/{config.train.epochs} "
            f"| total={avg_losses['loss/total']:.4f} "
            f"| recon={avg_losses['loss/recon']:.4f} "
            f"| latent={avg_losses['loss/latent']:.4f} "
            f"| tf={tf_ratio:.2f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| {epoch_time:.1f}s"
        )

        for k, v in avg_losses.items():
            writer.add_scalar(f"epoch/{k}", v, epoch)

        # ─── Validation ─────────────────────────────────
        if val_loader and (epoch + 1) % config.train.eval_interval == 0:
            model.eval()
            val_loss_total = 0.0
            val_latent_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    context = batch["context"].to(device)
                    target = batch["target"].to(device)

                    output = model(
                        context_frames=context,
                        target_frames=target,
                        teacher_forcing_ratio=0.0,
                    )

                    loss, components = criterion(
                        predicted_frames=output["predicted_frames"],
                        target_frames=target,
                        predicted_latents=output["predicted_latents"],
                        target_latents=output["target_latents"],
                    )

                    val_loss_total += components["loss/total"]
                    val_latent_loss += components["loss/latent"]
                    val_batches += 1

            val_avg_total = val_loss_total / max(val_batches, 1)
            val_avg_latent = val_latent_loss / max(val_batches, 1)

            writer.add_scalar("val/loss_total", val_avg_total, epoch)
            writer.add_scalar("val/loss_latent", val_avg_latent, epoch)

            print(f"  Val: total={val_avg_total:.4f}, latent={val_avg_latent:.4f}")

            # Save best model
            if val_avg_latent < best_val_loss:
                best_val_loss = val_avg_latent
                patience_counter = 0
                save_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_avg_latent,
                    "config": {
                        "d_model": config.model.d_model,
                        "n_layers": config.model.n_layers,
                        "n_heads": config.model.n_heads,
                        "d_ff": config.model.d_ff,
                        "t_in": config.model.t_in,
                        "t_out": config.model.t_out,
                        "encoder_channels": config.model.encoder_channels,
                        "channels": config.data.channels,
                    },
                    "param_counts": param_counts,
                }, save_path)
                print(f"  [SAVED] Best model saved (val latent loss: {val_avg_latent:.4f})")
            else:
                patience_counter += 1

            # Generate sample GIFs
            if (epoch + 1) % config.train.save_interval == 0:
                gif_dir = os.path.join(config.train.log_dir, run_name, f"gifs_epoch_{epoch + 1}")
                evaluate_model(
                    model, val_loader, device=device,
                    num_samples=config.train.num_eval_samples,
                    output_dir=gif_dir,
                )

        # ─── Periodic checkpoint ────────────────────────
        if (epoch + 1) % config.train.save_interval == 0:
            save_path = os.path.join(
                config.train.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            }, save_path)

        # ─── Early stopping ─────────────────────────────
        if patience_counter >= config.train.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement.")
            break

    writer.close()
    print(f"\nTraining complete! Best validation latent loss: {best_val_loss:.4f}")
    print(f"Logs: {os.path.join(config.train.log_dir, run_name)}")
    print(f"Best model: {os.path.join(config.train.checkpoint_dir, 'best_model.pt')}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Latent Video Predictor")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.no_amp:
        config.train.use_amp = False

    train(config)
