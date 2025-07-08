import torch
from models.engine import Engine
from config.config import Config
import constants as cst

# === USER: Set these paths ===
OLD_CHECKPOINT_PATH = 'data/checkpoints/TLOB/HuggingFace/FI-2010_horizon_10_TLOB_seed_42.ckpt'  # Path to your old checkpoint
NEW_CHECKPOINT_PATH = 'data/checkpoints/TLOB/HuggingFace/FI-2010_horizon_10_TLOB_seed_42_converted.ckpt'  # Path to save the new checkpoint

# === USER: Set up your config as you would for evaluation ===
# You may need to import or construct your config here. This is a minimal example:
config = Config(
    model=None,  # Fill in as needed
    dataset=None,  # Fill in as needed
)

# === Build your current model as you would in your main code ===
# You may need to adjust these parameters to match your use case
model = Engine(
    seq_size=40,  # Example value
    horizon=10,   # Example value
    max_epochs=10,  # Example value
    model_type='TLOB',
    is_wandb=False,
    experiment_type=['EVALUATION'],
    lr=0.001,
    optimizer='Adam',
    dir_ckpt='converted.ckpt',
    hidden_dim=144,  # Example value
    num_layers=4,    # Example value
    num_features=40, # Example value
    dataset_type='FI_2010',
    num_heads=4,     # Example value
    is_sin_emb=True,
    num_classes=3,
    len_test_dataloader=100  # Example value
)

# === Load the old checkpoint ===
old_ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu')
if 'state_dict' in old_ckpt:
    old_state = old_ckpt['state_dict']
else:
    old_state = old_ckpt

# === Get the new model's state_dict ===
new_state = model.state_dict()

# === Attempt to match and copy weights ===
matched, skipped = 0, 0
for k in new_state.keys():
    # Try to find a matching key in the old checkpoint
    if k in old_state and old_state[k].shape == new_state[k].shape:
        new_state[k] = old_state[k]
        matched += 1
    else:
        skipped += 1
        # Optionally print skipped keys for debugging
        # print(f'Skipped: {k}')

print(f"Matched {matched} parameters, skipped {skipped}.")

# === Save the new checkpoint ===
torch.save({'state_dict': new_state}, NEW_CHECKPOINT_PATH)
print(f"Converted checkpoint saved to {NEW_CHECKPOINT_PATH}") 