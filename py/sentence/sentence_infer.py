import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

class SentenceBERTDataset(Dataset):
    def __init__(self, csv_file, test_mode=False, X_data=None, y_data=None):
        if test_mode:
            # For test mode, use passed data directly
            self.X = X_data.astype("float32")
            self.y = y_data.astype("int64") 
        else:
            # Read data from CSV file
            df = pd.read_csv(csv_file)
            # Extract q0 ~ q383 as features (384-dimensional sentence vectors)
            self.X = df[[f"q{i}" for i in range(384)]].values.astype("float32")
            # Extract labels (0: invalid, 1: valid)
            self.y = df["label"].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
class SentenceCNN(nn.Module):
    def __init__(self, input_dim=384, num_classes=1, dropout_rate=0.3):
        super(SentenceCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(dropout_rate)

    def forward(self, x):
        #print(f"[Input] x: {x.shape}")  # (batch, 384)
        x = x.unsqueeze(1)
        #print(f"After unsqueeze: {x.shape}")  # (batch, 1, 384)

        # Conv1
        x = F.relu(self.conv1(x))
        #print(f"After conv1: {x.shape}")  # (batch, 64, 384)
        x = self.dropout1(x)

        # Conv2
        x = F.relu(self.conv2(x))
        #print(f"After conv2: {x.shape}")  # (batch, 128, 384)
        x = self.dropout2(x)
        
        # Conv3
        x = F.relu(self.conv3(x))
        #print(f"After conv3: {x.shape}")  # (batch, 256, 384)
        x = self.dropout3(x)
        
        # AdaptiveMaxPool1d
        x = self.pool(x)
        #print(f"After AdaptiveMaxPool1d: {x.shape}")  # (batch, 256, 1)

        # Flatten
        x = x.squeeze(-1)
        #print(f"After squeeze: {x.shape}")  # (batch, 256)

        # FC1
        x = F.relu(self.fc1(x))
        #print(f"After fc1: {x.shape}")  # (batch, 128)
        x = self.dropout_fc(x)

        # FC2
        x = self.fc2(x)
        #print(f"After fc2: {x.shape}")  # (batch, 1)

        x = torch.sigmoid(x)
        #print(f"[Output] Final sigmoid: {x.shape}")
        return x

def load_data(csv_file, test_size=0.2, random_state=42):
    """
    Load and split data
    """
    df = pd.read_csv(csv_file)
    X = df[[f"q{i}" for i in range(384)]].values
    y = df["label"].values
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    """
    Create data loaders
    """
    train_dataset = SentenceBERTDataset(None, test_mode=True, X_data=X_train, y_data=y_train)
    test_dataset = SentenceBERTDataset(None, test_mode=True, X_data=X_test, y_data=y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-3, device='cuda', save_weights=True, weight_save_path='best_model_weights.pth'):
    """
    Train model with progress bars and automatic weight saving
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("Warning: tqdm not installed, using simple progress display")
        use_tqdm = False
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Create overall progress bar
    if use_tqdm:
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    else:
        epoch_pbar = range(num_epochs)
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create training batch progress bar
        if use_tqdm:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", 
                             leave=False, unit="batch")
        else:
            train_pbar = train_loader
            
        for batch_idx, (batch_x, batch_y) in enumerate(train_pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y.float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y.float()).sum().item()
            
            # Update training progress bar
            if use_tqdm and batch_idx % 10 == 0:
                current_acc = train_correct / train_total if train_total > 0 else 0
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
            elif not use_tqdm and batch_idx % max(1, len(train_loader)//10) == 0:
                current_acc = train_correct / train_total if train_total > 0 else 0
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"\r  Training Progress: {progress:.1f}% | Loss: {loss.item():.4f} | Acc: {current_acc:.4f}", end="")
        
        if not use_tqdm:
            print()  # New line
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        if use_tqdm:
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing", 
                            leave=False, unit="batch")
        else:
            test_pbar = test_loader
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(test_pbar):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y.float())
                
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y.float()).sum().item()
                
                # Update test progress bar
                if use_tqdm and batch_idx % 10 == 0:
                    current_acc = test_correct / test_total if test_total > 0 else 0
                    test_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.4f}'
                    })
                elif not use_tqdm and batch_idx % max(1, len(test_loader)//5) == 0:
                    current_acc = test_correct / test_total if test_total > 0 else 0
                    progress = (batch_idx + 1) / len(test_loader) * 100
                    print(f"\r  Testing Progress: {progress:.1f}% | Loss: {loss.item():.4f} | Acc: {current_acc:.4f}", end="")
        
        if not use_tqdm:
            print()  # New line
        
        # Calculate average loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            
            # Save weights immediately when we get a new best model
            if save_weights:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'best_test_acc': best_test_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies
                }, weight_save_path)
                
                if use_tqdm:
                    tqdm.write(f"💾 New best model saved! Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
                else:
                    print(f"💾 New best model saved! Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Calculate training time
        epoch_time = time.time() - start_time
        
        # Update overall progress bar
        if use_tqdm:
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'test_loss': f'{avg_test_loss:.4f}',
                'test_acc': f'{test_acc:.4f}',
                'best_acc': f'{best_test_acc:.4f}',
                'lr': f'{new_lr:.6f}',
                'time': f'{epoch_time:.1f}s'
            })
            
            # Display detailed information every 5 epochs
            if (epoch + 1) % 5 == 0:
                tqdm.write(f"\n📊 Epoch {epoch+1}/{num_epochs} Statistics:")
                tqdm.write(f"   Training: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
                tqdm.write(f"   Testing: Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}")
                tqdm.write(f"   Best Accuracy: {best_test_acc:.4f}")
                tqdm.write(f"   Learning Rate: {new_lr:.6f} {'(adjusted)' if new_lr != old_lr else ''}")
                tqdm.write(f"   Training Time: {epoch_time:.1f} seconds")
                tqdm.write("-" * 50)
        else:
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Complete:")
            print(f"   Training: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"   Testing: Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}")
            print(f"   Best Accuracy: {best_test_acc:.4f}")
            print(f"   Learning Rate: {new_lr:.6f} {'(adjusted)' if new_lr != old_lr else ''}")
            print(f"   Training Time: {epoch_time:.1f} seconds")
            print("-" * 50)
    
    if use_tqdm:
        epoch_pbar.close()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final save with complete training information
    if save_weights:
        final_save_path = weight_save_path.replace('.pth', '_final.pth')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'model_config': {
                'input_dim': 384,
                'num_classes': 1,
                'dropout_rate': getattr(model, 'dropout1', None).p if hasattr(model, 'dropout1') else 0.3
            },
            'training_config': {
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': train_loader.batch_size,
            },
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'final_train_acc': train_accuracies[-1] if train_accuracies else 0,
            'final_test_acc': test_accuracies[-1] if test_accuracies else 0
        }, final_save_path)
        
        print(f"💾 Final model saved to: {final_save_path}")
        print(f"💾 Best weights saved to: {weight_save_path}")
    
    print(f"\n🎉 Training Complete! Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model performance with progress bars
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("🔍 Evaluating model...")
    
    if use_tqdm:
        pbar = tqdm(test_loader, desc="Evaluation Progress", unit="batch")
    else:
        pbar = test_loader
        total_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            if not use_tqdm and batch_idx % max(1, total_batches//10) == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"\rEvaluation Progress: {progress:.1f}%", end="")
    
    if not use_tqdm:
        print("\nEvaluation Complete!")
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"📈 Test Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Invalid Sentence', 'Valid Sentence']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Invalid Sentence', 'Valid Sentence'],
                yticklabels=['Invalid Sentence', 'Valid Sentence'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("💾 Confusion matrix saved as confusion_matrix.png")
    plt.show()
    
    return accuracy

def load_model_weights(model, weight_path, device='cuda', load_optimizer=False, load_scheduler=False):
    """
    Load model weights from saved checkpoint
    
    Args:
        model: Model instance to load weights into
        weight_path: Path to the saved weights file
        device: Device to load the model on
        load_optimizer: Whether to return optimizer state dict
        load_scheduler: Whether to return scheduler state dict
    
    Returns:
        model: Model with loaded weights
        info: Dictionary containing training information
        optimizer_state: Optimizer state dict (if load_optimizer=True)
        scheduler_state: Scheduler state dict (if load_scheduler=True)
    """
    print(f"📂 Loading model weights from: {weight_path}")
    
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Extract training information
        info = {
            'best_epoch': checkpoint.get('best_epoch', checkpoint.get('epoch', 'Unknown')),
            'best_test_acc': checkpoint.get('best_test_acc', 'Unknown'),
            'final_train_acc': checkpoint.get('final_train_acc', 'Unknown'),
            'final_test_acc': checkpoint.get('final_test_acc', 'Unknown'),
            'train_losses': checkpoint.get('train_losses', []),
            'test_losses': checkpoint.get('test_losses', []),
            'train_accuracies': checkpoint.get('train_accuracies', []),
            'test_accuracies': checkpoint.get('test_accuracies', [])
        }
        
        print(f"✅ Model weights loaded successfully!")
        print(f"   Best Epoch: {info['best_epoch']}")
        print(f"   Best Test Accuracy: {info['best_test_acc']}")
        
        result = [model, info]
        
        if load_optimizer:
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            result.append(optimizer_state)
            print(f"   Optimizer state loaded: {'Yes' if optimizer_state else 'No'}")
        
        if load_scheduler:
            scheduler_state = checkpoint.get('scheduler_state_dict', None)
            result.append(scheduler_state)
            print(f"   Scheduler state loaded: {'Yes' if scheduler_state else 'No'}")
        
        return result if len(result) > 2 else (result[0], result[1])
        
    except FileNotFoundError:
        print(f"❌ Error: Weight file not found: {weight_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        raise

def save_model_weights(model, save_path, epoch=None, test_acc=None, train_acc=None, 
                      optimizer=None, scheduler=None, train_losses=None, test_losses=None,
                      train_accuracies=None, test_accuracies=None, additional_info=None):
    """
    Save model weights with training information
    
    Args:
        model: Model to save
        save_path: Path to save the weights
        epoch: Current epoch
        test_acc: Test accuracy
        train_acc: Training accuracy
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        train_losses: List of training losses
        test_losses: List of test losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
        additional_info: Additional information to save
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 384,
            'num_classes': 1,
            'dropout_rate': getattr(model, 'dropout1', None).p if hasattr(model, 'dropout1') else 0.3
        }
    }
    
    # Add training information if provided
    if epoch is not None:
        save_dict['epoch'] = epoch
    if test_acc is not None:
        save_dict['best_test_acc'] = test_acc
    if train_acc is not None:
        save_dict['final_train_acc'] = train_acc
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    if train_losses is not None:
        save_dict['train_losses'] = train_losses
    if test_losses is not None:
        save_dict['test_losses'] = test_losses
    if train_accuracies is not None:
        save_dict['train_accuracies'] = train_accuracies
    if test_accuracies is not None:
        save_dict['test_accuracies'] = test_accuracies
    if additional_info is not None:
        save_dict.update(additional_info)
    
    torch.save(save_dict, save_path)
    print(f"💾 Model weights saved to: {save_path}")

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(test_accuracies, label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_sentence(model, sentence_vector, device='cuda'):
    """
    Predict single sentence vector
    """
    model.eval()
    with torch.no_grad():
        # Convert sentence vector to tensor and add batch dimension
        x = torch.tensor(sentence_vector, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(x).squeeze().item()
        prediction = 1 if output > 0.5 else 0
        confidence = output if prediction == 1 else 1 - output
        
        return prediction, confidence

