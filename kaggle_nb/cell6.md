# Cell 6: Initialize the SegNext model for ISIC dataset with CrossNeXt decoder

# Initialize SegNext model with appropriate parameters for ISIC dataset
# The model will use CrossNeXt decoder due to our changes in model.py
model = SegNext(
    num_classes=config['num_classes'],       # 2 for binary segmentation (background and lesion)
    in_channnels=config['input_channels'],   # 3 for RGB images
    embed_dims=[32, 64, 160, 256],           # Reduced dimensions for Kaggle
    ffn_ratios=[4, 4, 4, 4],                 # Feed-forward network ratios
    depths=[3, 2, 2, 2],                     # Reduced depth for faster training
    num_stages=4,                            # Standard 4-stage architecture
    dec_outChannels=128,                     # Reduced decoder channels for Kaggle
    drop_path=float(config['stochastic_drop_path']),
    config=config
)

# Move model to available device (GPU if available)
model = model.to(device)

# Print model summary
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Input channels: {config['input_channels']}")
print(f"Output classes: {config['num_classes']}")
print(f"Using CrossNeXt decoder with {config.get('crossnext_num_heads', 8)} attention heads")

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
print("CrossNeXt decoder is being used for enhanced segmentation performance.") 