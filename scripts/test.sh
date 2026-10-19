#!/bin/bash
python -m src.test "checkpointing.load=${1:-checkpoints/CGR_thuman2.0.ckpt}" test.save_images=${2:-false} "${@:3}"
