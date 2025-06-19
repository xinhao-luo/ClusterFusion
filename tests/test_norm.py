import torch
import flashinfer
import clusterfusion

batch_size = 64
hidden_size = 8192

def test_norm():
    x = torch.randn(batch_size, hidden_size).to(0).half()
    w = torch.randn(hidden_size).to(0).half()

    y_gt = flashinfer.norm.rmsnorm(x, w)
    y = clusterfusion.rmsnorm(x, w)

    mae = (y - y_gt).abs().mean()
    print("Mean Absolute Error (MAE):", mae.item())

    mse = ((y - y_gt) ** 2).mean()
    print("Mean Squared Error (MSE):", mse.item())

    max_error = (y - y_gt).abs().max()
    print("Max Error:", max_error.item())

if __name__ == "__main__":
    test_norm()