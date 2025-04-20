# Cell 6: Initialize the SegNext model for ISIC dataset with CrossNext decoder

# Initialize SegNext model with parameters from config
model = SegNext(
    num_classes=config['num_classes'],       # 2 for binary segmentation (background and lesion)
    in_channnels=config['input_channels'],   # 3 for RGB images
    embed_dims=config['embed_dims'],         # [32, 64, 160, 256] for Tiny variant
    ffn_ratios=[4, 4, 4, 4],                 # Feed-forward network ratios
    depths=config['depths'],                 # [3, 3, 5, 2] for Tiny variant
    num_stages=4,                            # Standard 4-stage architecture
    dec_outChannels=config['decoder_channels'], # 64 for Tiny variant
    drop_path=float(config['stochastic_drop_path']),
    config=config
)

# Move model to available device (GPU if available)
model = model.to(device)

# Print model summary
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Input channels: {config['input_channels']}")
print(f"Output classes: {config['num_classes']}")
print(f"Using CrossNext decoder with {config['crossnext_num_heads']} attention heads")
print(f"Encoder dimensions: {config['embed_dims']}")
print(f"Encoder depths: {config['depths']}")
print(f"Decoder channels: {config['decoder_channels']}")

# Initialize loss function
loss = FocalLoss()
criterion = lambda x, y: loss(x, y)

# Initialize optimizer (AdamW for better regularization)
optimizer = torch.optim.AdamW(
    [{'params': model.parameters(), 'lr': config['learning_rate']}],
    weight_decay=config['WEIGHT_DECAY']
)

# Initialize learning rate scheduler
scheduler = LR_Scheduler(
    config['lr_schedule'],
    config['learning_rate'],
    config['epochs'],
    iters_per_epoch=len(train_loader),
    warmup_epochs=config['warmup_epochs']
)

# Initialize evaluation metrics
metric = ConfusionMatrix(config['num_classes'])

# Initialize model utils for checkpoint management
mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])

# Initialize trainer and evaluator
trainer = Trainer(model, config['batch_size'], optimizer, criterion, metric)
evaluator = Evaluator(model, metric)

print("Model, optimizer, and training utilities initialized successfully.")
print("CrossNext decoder is being used for enhanced segmentation performance.") 