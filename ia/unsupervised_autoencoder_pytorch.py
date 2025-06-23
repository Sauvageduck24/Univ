import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Datos sintéticos (clusters)
X, _ = make_blobs(n_samples=1000, n_features=10, centers=3, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)

# Definir autoencoder simple
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.ReLU(),
            nn.Linear(6, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

model = Autoencoder(input_dim=10, latent_dim=2)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    X_rec = model(X)
    loss = loss_fn(X_rec, X)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Obtener representaciones latentes
model.eval()
with torch.no_grad():
    Z = model.encoder(X)
    print("Primeras 5 representaciones latentes (2D):\n", Z[:5]) 