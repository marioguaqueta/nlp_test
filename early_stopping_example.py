"""
Add this to your train.py or train_resume.py to implement early stopping
"""

from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        """
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0
        self.wait = 0
        
    def on_epoch_end(self, args, state, control, **kwargs):
        # Get current F1 from logs
        current_f1 = state.log_history[-1].get('eval/f1_score', 0)
        
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.wait = 0
            print(f"✓ New best F1: {current_f1:.4f}")
        else:
            self.wait += 1
            print(f"⚠️ No improvement for {self.wait} epochs (best: {self.best_f1:.4f})")
            
            if self.wait >= self.patience:
                print(f"🛑 Early stopping triggered! Best F1: {self.best_f1:.4f}")
                control.should_training_stop = True
                
        return control

# Add to your trainer:
early_stop = EarlyStoppingCallback(patience=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
    callbacks=[eval_callback, early_stop]  # Add early stopping
)
