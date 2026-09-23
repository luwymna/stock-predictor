import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#config
TICKER = "AAPL"
PERIOD = "5y"
LOOKBACK = 60
EPOCHS = 150          
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#fetch data
def fetch_data(ticker, period="5y"):
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if len(df) < LOOKBACK + 100:
        raise ValueError(f"Not enough data: {len(df)} rows.")

    print(f"Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    return df


#preprocess
def preprocess(df, lookback, train_split):
    values = df.values.astype(np.float32)
    print(f"DEBUG: values shape = {values.shape}")

    if len(values) <= lookback:
        raise ValueError(f"Not enough rows ({len(values)}) for lookback {lookback}.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    print(f"DEBUG: scaled shape = {scaled.shape}, lookback = {lookback}")

    close_idx = list(df.columns).index("Close")

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, close_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"DEBUG: X shape = {X.shape}, y shape = {y.shape}")

    if len(X) < 10:
        raise ValueError(f"Too few samples after windowing: {len(X)}.")

    split = int(len(X) * train_split)
    if split == 0 or split == len(X):
        raise ValueError(f"Bad split: train={split}, total={len(X)}.")

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, y_test, scaler, close_idx


#model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        last_close = x[:, -1, 3:4]
        return last_close + self.fc(out)


#train
def train_model(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, lr):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(test_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Track the best state for shipping
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

    # Restore the best-seen weights (early stopping bonus)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model with val loss = {best_val:.6f}")

    return train_losses, val_losses


#evaluate
def evaluate(model, X_test, y_test, scaler, close_idx, n_features):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).squeeze(-1).cpu().numpy()

    def inverse_close(values):
        dummy = np.zeros((len(values), n_features), dtype=np.float32)
        dummy[:, close_idx] = values
        return scaler.inverse_transform(dummy)[:, close_idx]

    preds_real = inverse_close(preds)
    actual_real = inverse_close(y_test)

    rmse = float(np.sqrt(np.mean((preds_real - actual_real) ** 2)))
    mae = float(np.mean(np.abs(preds_real - actual_real)))
    mape = float(np.mean(np.abs((preds_real - actual_real) / actual_real)) * 100)

    print(f"\nMetrics")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    return preds_real, actual_real, mape


#naive baseline (tomorrow = today)
def naive_baseline(df, lookback, train_split, actual_prices, model_mape):
    closes = df["Close"].values.astype(np.float32)
    n_samples = len(closes) - lookback
    split = int(n_samples * train_split)
    test_start = lookback + split

    naive_preds = closes[test_start - 1 : test_start - 1 + len(actual_prices)]
    actual = closes[test_start : test_start + len(actual_prices)]

    assert len(naive_preds) == len(actual), \
        f"Length mismatch: naive={len(naive_preds)}, actual={len(actual)}"

    naive_rmse = float(np.sqrt(np.mean((naive_preds - actual) ** 2)))
    naive_mape = float(np.mean(np.abs((naive_preds - actual) / actual)) * 100)

    print(f"\n--- Naive Baseline (tomorrow = today) ---")
    print(f"RMSE: {naive_rmse:.4f}")
    print(f"MAPE: {naive_mape:.2f}%")

    print(f"\n--- Model vs Baseline ---")
    print(f"Model MAPE: {model_mape:.2f}%")
    print(f"Naive MAPE: {naive_mape:.2f}%")
    delta = naive_mape - model_mape
    if delta > 0:
        print(f"Model beats baseline by {delta:.2f}% MAPE ✅")
    else:
        print(f"Model is WORSE than baseline by {abs(delta):.2f}% MAPE ⚠️")


#plot
def plot_results(train_losses, val_losses, preds, actual, ticker):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_title("Training Loss (150 epochs)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(actual, label="Actual", alpha=0.8)
    axes[1].plot(preds, label="Predicted", alpha=0.8)
    axes[1].set_title(f"{ticker} — Predicted vs Actual (Test Set)")
    axes[1].set_xlabel("Time (test samples)")
    axes[1].set_ylabel("Price ($)")
    axes[1].legend()

    plt.tight_layout()
    out_path = f"{ticker}_prediction.png"
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"\nSaved plot to {out_path}")


#main
if __name__ == "__main__":
    df = fetch_data(TICKER, PERIOD)
    X_train, y_train, X_test, y_test, scaler, close_idx = preprocess(
        df, LOOKBACK, TRAIN_SPLIT
    )

    n_features = X_train.shape[2]
    model = StockLSTM(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LEARNING_RATE,
    )

    preds, actual, model_mape = evaluate(
        model, X_test, y_test, scaler, close_idx, n_features
    )

    naive_baseline(df, LOOKBACK, TRAIN_SPLIT, actual, model_mape)

    plot_results(train_losses, val_losses, preds, actual, TICKER)

    torch.save(model.state_dict(), f"{TICKER}_lstm.pt")
    print(f"\nModel saved to {TICKER}_lstm.pt")