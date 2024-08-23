from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, test_loader, device):
    model.eval()
    mse_list = []
    mae_list = []
    
    with torch.no_grad():
        for L, ab, user_hints in test_loader:
            L, ab, user_hints = L.to(device), ab.to(device), user_hints.to(device)
            outputs = model(L, user_hints)
            
            mse = mean_squared_error(ab.cpu().numpy(), outputs.cpu().numpy())
            mae = mean_absolute_error(ab.cpu().numpy(), outputs.cpu().numpy())
            
            mse_list.append(mse)
            mae_list.append(mae)
    
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    
    print(f"Average MSE: {avg_mse}")
    print(f"Average MAE: {avg_mae}")
    
    return avg_mse, avg_mae

# After training, evaluate the model
test_dataset = ColorizationDataset(root_dir='path/to/your/test/images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

avg_mse, avg_mae = evaluate_model(model, test_loader, device)
