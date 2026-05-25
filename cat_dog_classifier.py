import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class CatDogClassificationDataset(Dataset):
    def __init__(self, base_folder):
        self.cat_paths = glob.glob(os.path.join(base_folder, "Cat", "*.jpg"))
        self.dog_paths = glob.glob(os.path.join(base_folder, "Dog", "*.jpg"))
        
        # 0.0 = Cat, 1.0 = Dog
        self.all_samples = [(p, 0.0) for p in self.cat_paths] + [(p, 1.0) for p in self.dog_paths]
        
    def __len__(self):
        return len(self.all_samples)
        
    def __getitem__(self, idx):
        img_path, label = self.all_samples[idx]
        img = cv2.imread(img_path)
        
        while img is None:
            idx = (idx + 1) % len(self.all_samples)
            img_path, label = self.all_samples[idx]
            img = cv2.imread(img_path)
            
        # dopasuj param do autoencodera
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return tensor_img, torch.tensor([label], dtype=torch.float32)
    
from cat_autoencoder import CatAutoencoder

class CatDogClassifier(nn.Module):
    def __init__(self, pretrained_ae_path):
        super(CatDogClassifier, self).__init__()
        
        # ladujemy autoenkoder
        base_ae = CatAutoencoder()
        base_ae.load_state_dict(torch.load(pretrained_ae_path, map_location=torch.device('cpu')))
        
        # nie trenujemy autoenkodera dlatego freeze
        self.encoder = base_ae.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # dodajemy layers do klasyfikatora
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
    
            nn.Linear(128, 32),
            nn.ReLU(),
            
            # output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        # wez feature i klasyfikuj
        features = self.encoder(x)
        probability = self.classifier(features)
        return probability
    
def train_classifier(dataset_path, ae_weights_path, save_classifier_path="classifier_v1.pth", fine_tune = False, epochs=10):
    print(f"Training classifier")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    # inicjalizuj 
    model = CatDogClassifier(ae_weights_path).to(device)
    dataset = CatDogClassificationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    criterion = nn.BCELoss()
    # optymalizujemy klasyfikator, nie autoenkoder
    if fine_tune:
        for param in model.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    loss_history = []
    acc_history = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # obecne paramentry
            running_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    torch.save(model.state_dict(), save_classifier_path)
    print(f"Classifier saved successfully to {save_classifier_path}")
    return model, loss_history, acc_history

def predict_image(image_path, trained_classifier_path):
    model = CatDogClassifier("cat_autoencoder_v1.pth") 
    model.load_state_dict(torch.load(trained_classifier_path, map_location=torch.device('cpu')))
    model.eval()
    
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"
        
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor_img = tensor_img.unsqueeze(0)
    
    with torch.no_grad():
        probability = model(tensor_img).item()
        
    if probability > 0.5:
        return f"DOG ({probability * 100:.2f}% confidence)"
    else:
        return f"CAT ({(1 - probability) * 100:.2f}% confidence)"
    
import csv
import matplotlib.pyplot as plt

def save_experiments_to_csv(experiments_dict, csv_filename="all_experiments_results.csv"):
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Epoch", "Loss", "Accuracy"])
        
        for exp_name, metrics in experiments_dict.items():
            losses = metrics["loss"]
            accuracies = metrics["acc"]
            
            for i in range(len(losses)):
                writer.writerow([
                    exp_name, 
                    i + 1, 
                    round(losses[i], 4), 
                    round(accuracies[i], 2)
                ])
                
    print(f"Successfully wrote all experimental logs to: {csv_filename}")


def plot(experiments_dict, loss_filename="loss_comparison.png", acc_filename="accuracy_comparison.png"):
    plt.figure(figsize=(8, 5))
    
    for exp_name, metrics in experiments_dict.items():
        losses = metrics["loss"]
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, label=exp_name, marker="o", linewidth=2)
        
    plt.title("Loss Curve Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(loss_filename, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved loss chart to: {loss_filename}")

    plt.figure(figsize=(8, 5))
    
    for exp_name, metrics in experiments_dict.items():
        accuracies = metrics["acc"]
        epochs = list(range(1, len(accuracies) + 1))
        plt.plot(epochs, accuracies, label=exp_name, marker="s", linewidth=2)
        
    plt.title("Accuracy Curve Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.ylim(40, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(acc_filename, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved accuracy chart to: {acc_filename}")


if __name__ == "__main__":
    all_runs = {}

    a, loss1, acc1 = train_classifier("PetImages", "cat_autoencoder_v1.pth", fine_tune=False, save_classifier_path="cat_dog_classifier_v1_nft.pth", epochs=100)

    all_runs["Frozen Encoder"] = {"loss": loss1, "acc": acc1}

    a, loss2, acc2 = train_classifier("PetImages", "cat_autoencoder_v1.pth", fine_tune=True, save_classifier_path="cat_dog_classifier_v1_ft.pth", epochs=100)

    all_runs["Unfrozen Encoder"] = {"loss": loss2, "acc": acc2}

    save_experiments_to_csv(all_runs, csv_filename="nn_experiment_logs.csv")
    plot(
        experiments_dict=all_runs, 
        loss_filename="chart_loss_comparison.png", 
        acc_filename="chart_accuracy_comparison.png"
    )

