import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import TimesformerModel, TimesformerConfig
from torchvision import transforms
from tqdm import tqdm

# Configuration
DATASET_PATH = '/Users/aks/Desktop/thesis/PHYIO-DATASET'
NUM_FRAMES = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
BATCH_SIZE = 8  # Reduced batch size for TimeSformer as it requires more memory
EPOCHS = 40
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning pre-trained model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Define the exercise types and sides
exercise_types = [
    'Arm_Circumduction', 'Shoulder_Abduction', 'Shoulder_Flexion', 
    'Wrist-extension-stretch', 'ankle', 'ballpress', 
    'cross-body_shoulder_stretch', 'isoetric_rotation', 'isoetric_side_bending'
]
sides = ['LEFT', 'RIGHT', 'NONE']  # NONE for exercises like ankle that don't have side labels

def get_labels_from_path(video_path):
    """Extract exercise type and side from the video path"""
    filename = os.path.basename(video_path)
    parent_dir = os.path.basename(os.path.dirname(video_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    
    if grandparent_dir in exercise_types:
        exercise_type = grandparent_dir
        if parent_dir in sides:
            side = parent_dir
        else:
            side = 'NONE'
    else:
        # Handle the ankle case which has a different structure
        exercise_type = 'ankle'
        side = 'NONE'
    
    return exercise_type, side

def load_video_data():
    """Load videos and generate labels"""
    data = []
    
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                exercise_type, side = get_labels_from_path(video_path)
                data.append({
                    'video_path': video_path,
                    'exercise_type': exercise_type,
                    'side': side
                })
    
    return pd.DataFrame(data)

def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and convert to RGB
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # If we couldn't extract exactly num_frames, pad with zeros
    while len(frames) < num_frames:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
    
    return np.array(frames)

class PhysioVideoDataset(Dataset):
    """PyTorch Dataset for physiotherapy exercise videos"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
        # Map labels to indices
        self.exercise_type_to_idx = {ex: idx for idx, ex in enumerate(exercise_types)}
        self.side_to_idx = {side: idx for idx, side in enumerate(sides)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['video_path']
        
        # Extract frames
        frames = extract_frames(video_path)
        
        # Apply transforms if provided
        if self.transform:
            # Apply the same transform to each frame
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        else:
            # Convert numpy to torch tensor
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Get labels
        exercise_type = self.df.iloc[idx]['exercise_type']
        side = self.df.iloc[idx]['side']
        
        exercise_label = self.exercise_type_to_idx[exercise_type]
        side_label = self.side_to_idx[side]
        
        return frames, exercise_label, side_label

class TimeSformerForMultiTaskClassification(nn.Module):
    """TimeSformer model adapted for multi-task classification"""
    def __init__(self, num_exercise_classes, num_side_classes):
        super(TimeSformerForMultiTaskClassification, self).__init__()
        
        # Initialize TimeSformer with pretrained weights
        # Note: We don't modify num_frames in config to ensure compatibility with pre-trained weights
        self.config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.timesformer = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        
        # Freeze most of the TimeSformer layers for fine-tuning
        # We'll only train the last transformer blocks and classification layers
        for name, param in self.timesformer.named_parameters():
            # Only fine-tune the last 2 transformer blocks
            if not any(f'encoder.layer.{i}.' in name for i in range(10, 12)):
                param.requires_grad = False
        
        # Multi-task classification heads
        hidden_size = self.timesformer.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        
        # Exercise type classification head
        self.exercise_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_exercise_classes)
        )
        
        # Side classification head
        self.side_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_side_classes)
        )
    
    def forward(self, x):
        # TimeSformer expects input shape: (batch_size, num_frames, num_channels, height, width)
        # Rearrange to meet TimeSformer's expected input shape
        batch_size, num_frames, channels, height, width = x.shape
        
        # Forward pass through TimeSformer
        outputs = self.timesformer(x)
        
        # Get the [CLS] token output
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)
        
        # Classification heads
        exercise_logits = self.exercise_classifier(cls_token)
        side_logits = self.side_classifier(cls_token)
        
        return exercise_logits, side_logits

def train_epoch(model, dataloader, optimizer, criterion_exercise, criterion_side, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    exercise_correct = 0
    side_correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for frames, exercise_labels, side_labels in progress_bar:
        # Move data to device
        frames = frames.to(device)
        exercise_labels = exercise_labels.to(device)
        side_labels = side_labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        exercise_logits, side_logits = model(frames)
        
        # Calculate loss
        loss_exercise = criterion_exercise(exercise_logits, exercise_labels)
        loss_side = criterion_side(side_logits, side_labels)
        loss = loss_exercise + loss_side
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate accuracy
        _, exercise_preds = torch.max(exercise_logits, 1)
        _, side_preds = torch.max(side_logits, 1)
        exercise_correct += (exercise_preds == exercise_labels).sum().item()
        side_correct += (side_preds == side_labels).sum().item()
        total += exercise_labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'ex_acc': 100 * exercise_correct / total,
            'side_acc': 100 * side_correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    exercise_acc = 100 * exercise_correct / total
    side_acc = 100 * side_correct / total
    
    return epoch_loss, exercise_acc, side_acc

def validate(model, dataloader, criterion_exercise, criterion_side, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    exercise_correct = 0
    side_correct = 0
    total = 0
    
    y_true_exercise = []
    y_pred_exercise = []
    y_true_side = []
    y_pred_side = []
    
    with torch.no_grad():
        for frames, exercise_labels, side_labels in tqdm(dataloader, desc="Validating"):
            # Move data to device
            frames = frames.to(device)
            exercise_labels = exercise_labels.to(device)
            side_labels = side_labels.to(device)
            
            # Forward pass
            exercise_logits, side_logits = model(frames)
            
            # Calculate loss
            loss_exercise = criterion_exercise(exercise_logits, exercise_labels)
            loss_side = criterion_side(side_logits, side_labels)
            loss = loss_exercise + loss_side
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy
            _, exercise_preds = torch.max(exercise_logits, 1)
            _, side_preds = torch.max(side_logits, 1)
            exercise_correct += (exercise_preds == exercise_labels).sum().item()
            side_correct += (side_preds == side_labels).sum().item()
            total += exercise_labels.size(0)
            
            # Store predictions for classification report
            y_true_exercise.extend(exercise_labels.cpu().numpy())
            y_pred_exercise.extend(exercise_preds.cpu().numpy())
            y_true_side.extend(side_labels.cpu().numpy())
            y_pred_side.extend(side_preds.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    exercise_acc = 100 * exercise_correct / total
    side_acc = 100 * side_correct / total
    
    return epoch_loss, exercise_acc, side_acc, y_true_exercise, y_pred_exercise, y_true_side, y_pred_side

def plot_training_history(train_losses, val_losses, train_exercise_accs, val_exercise_accs, 
                          train_side_accs, val_side_accs):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot exercise accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_exercise_accs, label='Train Exercise Accuracy')
    plt.plot(val_exercise_accs, label='Validation Exercise Accuracy')
    plt.title('Exercise Type Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot side accuracy
    plt.subplot(2, 2, 3)
    plt.plot(train_side_accs, label='Train Side Accuracy')
    plt.plot(val_side_accs, label='Validation Side Accuracy')
    plt.title('Side Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('timesformer_training_history.png')
    plt.show()

def evaluate_model(model, test_dataloader, device):
    """Evaluate the model on test data"""
    model.eval()
    
    all_exercise_preds = []
    all_exercise_labels = []
    all_side_preds = []
    all_side_labels = []
    
    with torch.no_grad():
        for frames, exercise_labels, side_labels in tqdm(test_dataloader, desc="Testing"):
            # Move data to device
            frames = frames.to(device)
            
            # Forward pass
            exercise_logits, side_logits = model(frames)
            
            # Get predictions
            _, exercise_preds = torch.max(exercise_logits, 1)
            _, side_preds = torch.max(side_logits, 1)
            
            # Store predictions and labels
            all_exercise_preds.extend(exercise_preds.cpu().numpy())
            all_exercise_labels.extend(exercise_labels.numpy())
            all_side_preds.extend(side_preds.cpu().numpy())
            all_side_labels.extend(side_labels.numpy())
    
    # Print classification reports
    print("Exercise Type Classification Report:")
    print(classification_report(
        all_exercise_labels, all_exercise_preds, 
        target_names=exercise_types
    ))
    
    print("\nSide Classification Report:")
    print(classification_report(
        all_side_labels, all_side_preds, 
        target_names=sides
    ))
    
    # Plot confusion matrices
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 1)
    cm_exercise = confusion_matrix(all_exercise_labels, all_exercise_preds)
    sns.heatmap(cm_exercise, annot=True, fmt='d', cmap='Blues', 
                xticklabels=exercise_types, yticklabels=exercise_types)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Exercise Type Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    cm_side = confusion_matrix(all_side_labels, all_side_preds)
    sns.heatmap(cm_side, annot=True, fmt='d', cmap='Blues',
                xticklabels=sides, yticklabels=sides)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Side Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('timesformer_confusion_matrices.png')
    plt.show()
    
    return all_exercise_preds, all_exercise_labels, all_side_preds, all_side_labels

def predict_single_video(model, video_path, device):
    """Make prediction on a single video"""
    model.eval()
    
    # Extract frames
    frames = extract_frames(video_path)
    
    # Convert to tensor
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Forward pass
        exercise_logits, side_logits = model(frames)
        
        # Get predictions
        _, exercise_pred = torch.max(exercise_logits, 1)
        _, side_pred = torch.max(side_logits, 1)
        
        # Get confidence scores
        exercise_probs = torch.nn.functional.softmax(exercise_logits, dim=1)
        side_probs = torch.nn.functional.softmax(side_logits, dim=1)
        
        predicted_exercise = exercise_types[exercise_pred.item()]
        predicted_side = sides[side_pred.item()]
        
        exercise_confidence = exercise_probs[0][exercise_pred].item() * 100
        side_confidence = side_probs[0][side_pred].item() * 100
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Predicted Exercise: {predicted_exercise} (Confidence: {exercise_confidence:.2f}%)")
    print(f"Predicted Side: {predicted_side} (Confidence: {side_confidence:.2f}%)")
    
    return predicted_exercise, predicted_side, exercise_confidence, side_confidence

def main():
    # Print library versions for debugging
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    print("Loading video data...")
    df = load_video_data()
    
    print(f"Total videos: {len(df)}")
    print(f"Exercise types: {df['exercise_type'].nunique()}")
    print(f"Sides: {df['side'].nunique()}")
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[['exercise_type', 'side']])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[['exercise_type', 'side']])
    
    print(f"Training videos: {len(train_df)}")
    print(f"Validation videos: {len(val_df)}")
    print(f"Testing videos: {len(test_df)}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = PhysioVideoDataset(train_df, transform=transform)
    val_dataset = PhysioVideoDataset(val_df, transform=transform)
    test_dataset = PhysioVideoDataset(test_df, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing TimeSformer model...")
    model = TimeSformerForMultiTaskClassification(
        num_exercise_classes=len(exercise_types),
        num_side_classes=len(sides)
    )
    model = model.to(DEVICE)
    
    # Define loss functions and optimizer
    criterion_exercise = nn.CrossEntropyLoss()
    criterion_side = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer which is better for transformer models
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
      
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_exercise_accs = []
    val_exercise_accs = []
    train_side_accs = []
    val_side_accs = []
    
    best_val_loss = float('inf')
    
    # Train the model
    print("Training TimeSformer model...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_exercise_acc, train_side_acc = train_epoch(
            model, train_dataloader, optimizer, criterion_exercise, criterion_side, DEVICE
        )
        
        # Validate
        val_loss, val_exercise_acc, val_side_acc, y_true_exercise, y_pred_exercise, y_true_side, y_pred_side = validate(
            model, val_dataloader, criterion_exercise, criterion_side, DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Exercise Acc: {train_exercise_acc:.2f}%, Train Side Acc: {train_side_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Exercise Acc: {val_exercise_acc:.2f}%, Val Side Acc: {val_side_acc:.2f}%")
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_exercise_accs.append(train_exercise_acc)
        val_exercise_accs.append(val_exercise_acc)
        train_side_accs.append(train_side_acc)
        val_side_accs.append(val_side_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), 'best_timesformer_model.pth')
            
        # Print classification reports for validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nValidation Exercise Classification Report:")
            print(classification_report(
                y_true_exercise, y_pred_exercise, 
                target_names=exercise_types
            ))
            
            print("\nValidation Side Classification Report:")
            print(classification_report(
                y_true_side, y_pred_side, 
                target_names=sides
            ))
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, 
        train_exercise_accs, val_exercise_accs,
        train_side_accs, val_side_accs
    )
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('best_timesformer_model.pth'))
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    all_exercise_preds, all_exercise_labels, all_side_preds, all_side_labels = evaluate_model(
        model, test_dataloader, DEVICE
    )
    
    # Calculate overall accuracy
    exercise_accuracy = np.mean(np.array(all_exercise_preds) == np.array(all_exercise_labels)) * 100
    side_accuracy = np.mean(np.array(all_side_preds) == np.array(all_side_labels)) * 100
    overall_accuracy = np.mean(
        (np.array(all_exercise_preds) == np.array(all_exercise_labels)) & 
        (np.array(all_side_preds) == np.array(all_side_labels))
    ) * 100
    
    print("\n===============================================")
    print("SUMMARY OF TEST RESULTS")
    print("===============================================")
    print(f"Total test videos: {len(test_df)}")
    print(f"Exercise classification accuracy: {exercise_accuracy:.2f}%")
    print(f"Side classification accuracy: {side_accuracy:.2f}%")
    print(f"Overall accuracy (both correct): {overall_accuracy:.2f}%")
    
    # Example prediction on a single video
    sample_video = test_df.iloc[0]['video_path']
    print("\nSample prediction:")
    predicted_exercise, predicted_side, _, _ = predict_single_video(model, sample_video, DEVICE)
    
    # Function to evaluate all test videos individually
    def evaluate_all_test_videos(model, test_df, device):
        results = []
        
        for idx, row in test_df.iterrows():
            video_path = row['video_path']
            ground_truth_exercise = row['exercise_type']
            ground_truth_side = row['side']
            
            print(f"\nVideo {idx+1}/{len(test_df)}:")
            predicted_exercise, predicted_side, exercise_confidence, side_confidence = predict_single_video(
                model, video_path, device
            )
            
            exercise_correct = predicted_exercise == ground_truth_exercise
            side_correct = predicted_side == ground_truth_side
            
            print(f"Ground Truth Exercise: {ground_truth_exercise} - {'✓' if exercise_correct else '✗'}")
            print(f"Ground Truth Side: {ground_truth_side} - {'✓' if side_correct else '✗'}")
            print("-" * 50)
            
            results.append({
                'video_path': video_path,
                'ground_truth_exercise': ground_truth_exercise,
                'ground_truth_side': ground_truth_side,
                'predicted_exercise': predicted_exercise,
                'predicted_side': predicted_side,
                'exercise_confidence': exercise_confidence,
                'side_confidence': side_confidence,
                'exercise_correct': exercise_correct,
                'side_correct': side_correct
            })
        
        return pd.DataFrame(results)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_timesformer_model.pth')
    print("Final model saved to 'final_timesformer_model.pth'")
    
    #Uncomment to run evaluation on all test videos
    print("\nEvaluating all test videos...")
    results_df = evaluate_all_test_videos(model, test_df, DEVICE)
    results_df.to_csv('timesformer_test_results.csv', index=False)
    print("Results saved to 'timesformer_test_results.csv'")

if __name__ == "__main__":
    
    main()
    
