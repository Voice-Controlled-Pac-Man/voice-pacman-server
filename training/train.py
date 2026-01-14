import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from datasets import BackgroundNoiseDataset, PacmanDataset
from model import BackgroundNoiseAugmentation, PacManCNN, SmartRandomShiftAugumentation

def train():
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 150
    PATIENCE = 10

    DATA_ROOT = "./data"
    DATA_URL = 'speech_commands_v0.02'
    
    wandb.init(
        project="voice-pacman",
        name="pacman-cnn-shifted-background-noise-light",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "data_root": DATA_ROOT,
            "data_url": DATA_URL,
        }
    )
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    device = torch.device("cpu")
    wandb.config.update({"device": str(device)})

    train_dataset = PacmanDataset(
        root=DATA_ROOT, 
        url=DATA_URL, 
        download=True, 
        subset='training',
    )

    val_dataset = PacmanDataset(
        root=DATA_ROOT, 
        url=DATA_URL, 
        download=True, 
        subset='validation',
    )

    test_dataset = PacmanDataset(
        root=DATA_ROOT, 
        url=DATA_URL, 
        download=True, 
        subset='testing',
    )

    background_noise_dataset = BackgroundNoiseDataset(
        root=DATA_ROOT,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PacManCNN().to(device)
    augment = BackgroundNoiseAugmentation(background_noise_dataset).to(device)
    shift_augment = SmartRandomShiftAugumentation().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    wandb.watch(model, log="all", log_freq=100)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {len(train_dataset)} files...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for waveforms, labels in tqdm(train_loader, desc="Training", total=len(train_loader)):
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            waveforms = augment(waveforms)
            waveforms = shift_augment(waveforms)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc="Validation", total=len(val_loader)):
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "patience_counter": patience_counter,
        })
        
        print(f'Epoch {epoch+1}/{EPOCHS} | '
              f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | '
              f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_pacman_model.pth')
            wandb.save('best_pacman_model.pth')
            print("Saved new best model.")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early Stopping! Training stopped.")
            break
    
    wandb.finish()

    correct = 0
    total = 0
    for waveforms, labels in test_loader:
        with torch.no_grad():
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    train()