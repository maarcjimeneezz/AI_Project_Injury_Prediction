#%%
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
dataset_dir = script_dir.parent / "Dataset"
models_dir = script_dir / "models"
results_dir = script_dir / "results"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

day_path = dataset_dir / "day_approach_maskedID_timeseries.csv"
week_path = dataset_dir / "week_approach_maskedID_timeseries.csv"

sequence_lengths = [15] 
hidden_sizes = [16,18,20,22,24,26,28,30,32]
seeds = [6]
patience = 20
epochs = 500
lr = 2e-4
batch_size = 512

day_df = pd.read_csv(day_path, parse_dates=['Date'])
week_df = pd.read_csv(week_path, parse_dates=['Date'])

base_features = [
	"Athlete ID",
	"Date",
	"injury",
	"nr. sessions",
	"total km",
	"km Z3-4",
	"km Z5-T1-T2",
	"km sprinting",
	"strength training",
	"hours alternative",
	"perceived exertion",
	"perceived trainingSuccess",
	"perceived recovery",
]

df = day_df[base_features].sort_values(by=["Athlete ID", "Date"])

df = df.sort_values(["Athlete ID", "Date"]).copy()
feature_cols = [c for c in df.columns if c not in ["injury", "Athlete ID", "Date"]]

results_file = results_dir / "gru_results.csv"

for sequence_length in sequence_lengths:
	for hidden_size in hidden_sizes:
		for seed in seeds:
			model_path = (
				models_dir
				/ f"GRU_model_sequence_length_{sequence_length}_hidden_size_{hidden_size}_seed_{seed}"
			)

			if model_path.exists():
				print(f"Skipping existing model: {model_path}")
				continue

			np.random.seed(seed)
			torch.manual_seed(seed)

			sequence_frames = []
			sequence_labels = []

			for _, athlete_df in df.groupby("Athlete ID", sort=False):
				athlete_df = athlete_df.sort_values("Date")
				athlete_features = athlete_df[feature_cols].values.astype(np.float32)
				athlete_labels = athlete_df["injury"].values.astype(np.float32)

				for start_idx in range(0, len(athlete_df) - sequence_length + 1, 1):
					end_idx = start_idx + sequence_length
					sequence_frames.append(athlete_features[start_idx:end_idx])
					sequence_labels.append(athlete_labels[end_idx - 1])

			X_all = np.asarray(sequence_frames, dtype=np.float32)
			y_all = np.asarray(sequence_labels, dtype=np.float32)

			X_train_raw, X_valtest_raw, y_train, y_valtest = train_test_split(
				X_all,
				y_all,
				test_size=0.2,
				random_state=seed,
				stratify=y_all if len(np.unique(y_all)) > 1 else None,
			)

			X_val_raw, X_test_raw, y_val, y_test = train_test_split(
				X_valtest_raw,
				y_valtest,
				test_size=0.5,
				random_state=seed,
				stratify=y_valtest if len(np.unique(y_valtest)) > 1 else None,
			)

			mu = X_train_raw.mean(axis=(0, 1), keepdims=True)
			sigma = X_train_raw.std(axis=(0, 1), keepdims=True)
			sigma[sigma == 0] = 1.0

			X_train = (X_train_raw - mu) / sigma
			X_val = (X_val_raw - mu) / sigma
			X_test = (X_test_raw - mu) / sigma

			X_train_t = torch.tensor(X_train, dtype=torch.float32)
			y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
			X_val_t = torch.tensor(X_val, dtype=torch.float32)
			y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
			X_test_t = torch.tensor(X_test, dtype=torch.float32)
			y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

			num_pos = float((y_train == 1).sum())
			num_neg = float((y_train == 0).sum())
			pos_weight_value = num_neg / max(num_pos, 1.0)
			print(f"Simple workflow - Positive class weight: {pos_weight_value:.4f}")

			train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
			val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=True)
			test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=True)

			class GRU(nn.Module):
				def __init__(self, input_size, hidden_size):
					super().__init__()
					self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
					self.fc = nn.Linear(hidden_size, 1)

				def forward(self, x):
					_, h_n = self.gru(x)
					logits = self.fc(h_n[-1])
					return logits

			input_size = X_train.shape[2]
			hidden_size = hidden_size
			model = GRU(input_size=input_size, hidden_size=hidden_size).to(device)
			criterion = nn.BCEWithLogitsLoss(
				pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
			)
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)

			def evaluate(loader, model, criterion, threshold=0.5):
				model.eval()
				total_loss = 0.0
				total_count = 0
				all_probs = []
				all_true = []

				with torch.no_grad():
					for xb, yb in loader:
						xb, yb = xb.to(device), yb.to(device)
						logits = model(xb)
						loss = criterion(logits, yb)
						probs = torch.sigmoid(logits)

						total_loss += loss.item() * xb.size(0)
						total_count += yb.size(0)
						all_probs.extend(probs.cpu().numpy().flatten().tolist())
						all_true.extend(yb.cpu().numpy().astype(int).flatten().tolist())

				avg_loss = total_loss / max(total_count, 1)
				preds = (np.array(all_probs) >= threshold).astype(int)
				true = np.array(all_true).astype(int)
				return avg_loss, true, np.array(all_probs), preds

			best_val_loss = float("inf")
			best_state_dict = copy.deepcopy(model.state_dict())
			patience_counter = 0

			print("\n========== Training Simple Workflow ==========")

			for epoch in range(epochs):
				model.train()
				train_loss, train_total = 0.0, 0

				for xb, yb in train_loader:
					xb, yb = xb.to(device), yb.to(device)

					optimizer.zero_grad()
					logits = model(xb)
					loss = criterion(logits, yb)
					loss.backward()
					optimizer.step()

					train_loss += loss.item() * xb.size(0)
					train_total += yb.size(0)

				train_loss /= max(train_total, 1)

				val_loss, _, _, _ = evaluate(val_loader, model, criterion, threshold=0.5)

				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_state_dict = copy.deepcopy(model.state_dict())
					patience_counter = 0
				else:
					patience_counter += 1

				print(
					f"Epoch [{epoch + 1}/{epochs}] ",
					f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{patience}",
				)

				if patience_counter >= patience:
					print(f"Early stopping triggered at epoch {epoch + 1}.")
					break

			model.load_state_dict(best_state_dict)
			torch.save(model.state_dict(), model_path)
			print(f"Best model saved as {model_path}")

			threshold_grid = np.linspace(0.01, 0.99, 99)
			_, y_val_true, y_val_probs, _ = evaluate(val_loader, model, criterion, threshold=0.5)
			val_balanced_accuracy_scores = []

			for threshold in threshold_grid:
				val_preds = (y_val_probs >= threshold).astype(int)
				val_balanced_accuracy_scores.append(balanced_accuracy_score(y_val_true, val_preds))

			best_threshold = threshold_grid[int(np.argmax(val_balanced_accuracy_scores))]

			test_loss, y_test_true, y_test_probs, y_test_pred = evaluate(
				test_loader, model, criterion, threshold=best_threshold
			)

			test_accuracy = accuracy_score(y_test_true, y_test_pred)
			test_balanced_accuracy = balanced_accuracy_score(y_test_true, y_test_pred)
			test_roc_auc = roc_auc_score(y_test_true, y_test_probs)

			results = {
				"seed": seed,
				"sequence_length": sequence_length,
				"hidden_size": hidden_size,
				"test_loss": test_loss,
				"test_accuracy": test_accuracy,
				"test_balanced_accuracy": test_balanced_accuracy,
				"test_roc_auc": test_roc_auc,
			}

			results_df = pd.DataFrame([results])
			if results_file.exists():
				results_df.to_csv(results_file, mode="a", header=False, index=False)
			else:
				results_df.to_csv(results_file, index=False)

			print(f"Results saved to {results_file}")


# %%
