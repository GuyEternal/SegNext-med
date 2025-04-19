# Create directories for checkpoints and logs
import os

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("Created checkpoint and log directories") 