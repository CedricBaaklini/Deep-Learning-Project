import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import psutil
from torch.cuda.amp import GradScaler, autocast

# Paths and global settings
PREPROCESSED_DATA_PATH = "/home/ah/Desktop/GNN/Preprocessed_Data/preprocessed_dataset.pkl"
NORMALIZATION_PARAMS_PATH = "/home/ah/Desktop/GNN/Preprocessed_Data/normalization_params.pkl"
MODEL_SAVE_PATH = "/home/ah/Desktop/GNN/gnn_atomic_charge_model.pth"
number_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed data
with open(PREPROCESSED_DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)
with open(NORMALIZATION_PARAMS_PATH, 'rb') as f:
    normalization_params = pickle.load(f)

print(f"Loaded {len(dataset)} graphs.")
print(f"Normalization params: {normalization_params}")

# GNN Model (simplified to avoid overfitting)
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim=5, edge_dim=11, hidden_dim=128, output_dim=1, heads=4):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3, edge_dim=edge_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()  # Output in [0, 1]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc(x)
        return self.sigmoid(x)

# Training function
def train_model(model, train_loader, val_loader, epochs, lr, accum_steps=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()
    scaler = GradScaler()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 20, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            with autocast():
                out = model(batch)
                loss = criterion(out, batch.y) / accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item() * accum_steps
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            del batch, out, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad(), autocast():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += criterion(out, batch.y).item()
                del batch, out
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        scheduler.step(val_loss)
        mem_usage = psutil.virtual_memory().percent
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}, Memory: {mem_usage:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace(".pth", "_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    test_loss, total_mae, ss_tot, ss_res, n_samples, mean_targets = 0, 0, 0, 0, 0, 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad(), autocast():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            test_loss += loss.item()
            preds = out.cpu().detach()
            targets = batch.y.cpu().detach()
            n_batch = targets.size(0)
            n_samples += n_batch
            total_mae += torch.sum(torch.abs(preds - targets)).item()
            mean_targets = (mean_targets * (n_samples - n_batch) + torch.sum(targets).item()) / n_samples
            ss_tot += torch.sum((targets - mean_targets) ** 2).item()
            ss_res += torch.sum((targets - preds) ** 2).item()
            del batch, out, loss, preds, targets
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        test_loss /= len(test_loader)

    mse = test_loss
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = total_mae / n_samples
    r2 = 1 - (ss_res / ss_tot)

    return test_loss, mse, rmse, mae, r2

# Main execution
def main():
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

    train_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(train_idx), num_workers=0)
    val_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(val_idx), num_workers=0)
    test_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(test_idx), num_workers=0)

    # Initialize model
    model = GNNModel(input_dim=5, edge_dim=11, hidden_dim=128, output_dim=1, heads=4).to(device)

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=number_epochs, lr=0.001)

    # Load best model and evaluate
    model.load_state_dict(torch.load(MODEL_SAVE_PATH.replace(".pth", "_best.pth")))
    test_loss, mse, rmse, mae, r2 = evaluate_model(model, test_loader)

    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
    plt.axhline(y=mse, color='r', linestyle='--', label=f"Test MSE = {mse:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ah/Desktop/GNN/loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
