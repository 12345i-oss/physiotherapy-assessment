#exercise type gets its own binary prediction (correct/incorrect)This uses multi-label classification with 17 binary outputs
#Each exercise type gets its own binary prediction (correct/incorrect)
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import VideoMAEModel, VideoMAEConfig, VideoMAEImageProcessor
from torchvision import transforms
from tqdm import tqdm
import json
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration - Keep your 32 frames
DATASET_PATH = '/Users/aks/Downloads/PhysiotherapyVideoOrganizer-1/PHYIO-DATASET'
LABELS_CSV_PATH = '/Users/aks/Desktop/label.csv' # Path to your labels CSV file
NUM_FRAMES = 32 # Keep your preferred frame count
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
BATCH_SIZE = 2 # Reduced for stability with larger input
EPOCHS = 40
LEARNING_RATE = 3e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpointsss'
RESUME_TRAINING = True

# Define the exercise types based on your directory structure
EXERCISE_TYPES = [
 'Arm_Circumduction_LEFT',
 'Arm_Circumduction_RIGHT',
 'Shoulder_Abduction_LEFT',
 'Shoulder_Abduction_RIGHT', 
 'Shoulder_Flexion_LEFT',
 'Shoulder_Flexion_RIGHT',
 'Wrist_extension_stretch_LEFT', 
 'Wrist_extension_stretch_RIGHT',
 'ankle',
 'ballpress_LEFT',
 'ballpress_RIGHT',
 'cross_body_shoulder_stretch_LEFT',
 'cross_body_shoulder_stretch_RIGHT',
 'isoetric_rotation_LEFT',
 'isoetric_rotation_RIGHT',
 'isoetric_side_bending_LEFT',
 'isoetric_side_bending_RIGHT',
]

NUM_EXERCISES = len(EXERCISE_TYPES)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Number of exercises: {NUM_EXERCISES}")
print(f"Number of frames per video: {NUM_FRAMES}")

def extract_exercise_type_from_path(video_path):
 """Extract exercise type from video path - Updated for directory structure"""
 path_parts = video_path.split(os.sep)
 
 # Look for the exercise pattern in the path
 for i, part in enumerate(path_parts):
 # Check for main exercise folders
 if 'Arm_Circumduction' in part:
 # Look for LEFT/RIGHT subfolder
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'Arm_Circumduction_{side}'
 
 if 'Shoulder_Abduction' in part:
 # Look for LEFT/RIGHT subfolder
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'Shoulder_Abduction_{side}'
 elif 'Shoulder_Flexion' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'Shoulder_Flexion_{side}'
 elif 'Wrist-extension-stretch' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'Wrist_extension_stretch_{side}'
 elif 'ballpress' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'ballpress_{side}'
 elif 'cross-body_shoulder_stretch' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'cross_body_shoulder_stretch_{side}'
 elif 'isoetric_rotation' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'isoetric_rotation_{side}'
 elif 'isoetric_side_bending' in part:
 if i + 1 < len(path_parts):
 side = path_parts[i + 1]
 if side in ['LEFT', 'RIGHT']:
 return f'isoetric_side_bending_{side}'
 elif 'Arm_Circumduction' in part:
 return 'Arm_Circumduction'
 elif 'ankle' in part.lower():
 return 'ankle'
 
 return None

def discover_exercises_from_dataset():
 """Discover all unique exercise types from the dataset structure"""
 discovered_exercises = set()
 
 if not os.path.exists(DATASET_PATH):
 print(f"Dataset path not found: {DATASET_PATH}")
 return []
 
 for root, dirs, files in os.walk(DATASET_PATH):
 for file in files:
 if file.endswith('.mp4'):
 video_path = os.path.join(root, file)
 exercise_type = extract_exercise_type_from_path(video_path)
 if exercise_type:
 discovered_exercises.add(exercise_type)
 
 discovered_list = sorted(list(discovered_exercises))
 print(f"Discovered exercises: {discovered_list}")
 return discovered_list

def load_video_data():
 """Load videos and their labels from CSV file or directory structure"""
 data = []
 missing_videos = []
 
 # First, let's discover what exercises actually exist in the dataset
 discovered_exercises = discover_exercises_from_dataset()
 
 if len(discovered_exercises) == 0:
 print("No exercises found in dataset!")
 return pd.DataFrame()
 
 if os.path.exists(LABELS_CSV_PATH):
 # Load from CSV if available
 print(f"Loading labels from CSV: {LABELS_CSV_PATH}")
 labels_df = pd.read_csv(LABELS_CSV_PATH)
 print(f"CSV columns: {labels_df.columns.tolist()}")
 
 for _, row in labels_df.iterrows():
 video_path = os.path.join(DATASET_PATH, row['video_path'])
 
 if os.path.exists(video_path):
 exercise_type = extract_exercise_type_from_path(video_path)
 if exercise_type:
 data.append({
 'video_path': video_path,
 'exercise_type': exercise_type,
 'correctness': int(row.get('label', row.get('correctness', 1))) # Default to correct if not specified
 })
 else:
 print(f"Could not extract exercise type from: {video_path}")
 else:
 missing_videos.append(video_path)
 else:
 # Load from directory structure
 print("CSV not found, loading from directory structure...")
 for root, dirs, files in os.walk(DATASET_PATH):
 for file in files:
 if file.endswith('.mp4'):
 video_path = os.path.join(root, file)
 exercise_type = extract_exercise_type_from_path(video_path)
 
 if exercise_type:
 # Default assumption: all videos are correct unless specified otherwise
 correctness = 1 # Assume correct for now
 
 data.append({
 'video_path': video_path,
 'exercise_type': exercise_type,
 'correctness': correctness
 })
 
 if missing_videos:
 print(f"Warning: {len(missing_videos)} videos not found")
 for vid in missing_videos[:5]: # Show first 5 missing videos
 print(f" Missing: {vid}")
 
 df = pd.DataFrame(data)
 
 if len(df) > 0:
 print(f"Loaded {len(df)} videos")
 print("Exercise distribution:")
 print(df['exercise_type'].value_counts())
 
 return df

def create_multi_label_targets(df):
 """Create multi-label targets for exercise classification"""
 # Update EXERCISE_TYPES to match discovered exercises
 global EXERCISE_TYPES, NUM_EXERCISES
 
 # Get unique exercise types from the dataframe
 unique_exercises = sorted(df['exercise_type'].unique())
 EXERCISE_TYPES = unique_exercises
 NUM_EXERCISES = len(EXERCISE_TYPES)
 
 print(f"Updated EXERCISE_TYPES: {EXERCISE_TYPES}")
 print(f"Updated NUM_EXERCISES: {NUM_EXERCISES}")
 
 # Create labels: for each video, we have binary labels (one for each exercise)
 multi_labels = []
 for _, row in df.iterrows():
 label_vector = np.zeros(NUM_EXERCISES, dtype=np.float32)
 
 # Find the index of the current exercise
 if row['exercise_type'] in EXERCISE_TYPES:
 exercise_idx = EXERCISE_TYPES.index(row['exercise_type'])
 # Set to 1 only if the exercise is performed correctly
 label_vector[exercise_idx] = float(row['correctness'])
 
 multi_labels.append(label_vector)
 
 return np.array(multi_labels)

def extract_frames(video_path, num_frames=NUM_FRAMES):
 """Extract frames from a video file"""
 frames = []
 cap = cv2.VideoCapture(video_path)
 
 if not cap.isOpened():
 print(f"Error: Could not open video {video_path}")
 return [np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) for _ in range(num_frames)]
 
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
 if total_frames <= 0:
 print(f"Error: Video has no frames {video_path}")
 cap.release()
 return [np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) for _ in range(num_frames)]
 
 frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
 
 for idx in frame_indices:
 cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
 ret, frame = cap.read()
 if ret:
 frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 frames.append(frame)
 else:
 print(f"Warning: Could not read frame {idx} from {video_path}")
 frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
 
 cap.release()
 
 # Pad with black frames if needed
 while len(frames) < num_frames:
 frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
 
 return frames

class PhysioVideoDataset(Dataset):
 """PyTorch Dataset for physiotherapy exercise videos - Multi-Label Classification"""
 def __init__(self, df, labels, processor, is_training=False):
 self.df = df
 self.labels = labels
 self.processor = processor
 self.is_training = is_training
 
 def __len__(self):
 return len(self.df)
 
 def __getitem__(self, idx):
 video_path = self.df.iloc[idx]['video_path']
 frames = extract_frames(video_path)
 
 # Apply simple augmentation for training data
 if self.is_training:
 if np.random.random() > 0.5:
 frames = [np.fliplr(frame) for frame in frames]
 
 # FIXED: Handle VideoMAE processor for 32 frames
 try:
 # Process frames directly without relying on VideoMAE's default 16-frame assumption
 # Convert to tensors manually
 frames_array = np.stack(frames) # Shape: (32, 224, 224, 3)
 
 # Normalize to [0, 1] and convert to tensor
 frames_tensor = torch.tensor(frames_array, dtype=torch.float32) / 255.0
 
 # Rearrange to (frames, channels, height, width)
 frames_tensor = frames_tensor.permute(0, 3, 1, 2) # (32, 3, 224, 224)
 
 except Exception as e:
 print(f"Error processing video {video_path}: {e}")
 frames_tensor = torch.zeros(NUM_FRAMES, 3, FRAME_HEIGHT, FRAME_WIDTH)
 
 label = torch.tensor(self.labels[idx], dtype=torch.float32)
 
 return frames_tensor, label

class VideoMAEForMultiLabelClassification(nn.Module):
 """VideoMAE model adapted for multi-label exercise classification with 32 frames"""
 def __init__(self, num_exercises):
 super(VideoMAEForMultiLabelClassification, self).__init__()
 
 # FIXED: Create custom config for 32 frames
 config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
 
 # Modify config for 32 frames
 config.num_frames = NUM_FRAMES # Set to 32
 
 # Load the pre-trained model with modified config
 self.videomae = VideoMAEModel.from_pretrained(
 "MCG-NJU/videomae-base-finetuned-kinetics",
 config=config,
 ignore_mismatched_sizes=True # IMPORTANT: This allows different frame counts
 )
 
 # Freeze most of the model parameters for fine-tuning
 for name, param in self.videomae.named_parameters():
 if not any(f'encoder.layer.{i}.' in name for i in range(10, 12)):
 param.requires_grad = False
 
 hidden_size = config.hidden_size
 self.dropout = nn.Dropout(0.1)
 
 # Multi-label classifier
 self.classifier = nn.Sequential(
 nn.Linear(hidden_size, 512),
 nn.ReLU(),
 nn.Dropout(0.1),
 nn.Linear(512, 256),
 nn.ReLU(),
 nn.Dropout(0.1),
 nn.Linear(256, num_exercises)
 )
 
 # FIXED: Handle temporal dimension adaptation
 self.temporal_adapter = nn.Conv1d(
 in_channels=config.hidden_size,
 out_channels=config.hidden_size,
 kernel_size=3,
 padding=1
 )
 
 def forward(self, pixel_values):
 # Ensure correct input shape
 if pixel_values.dim() == 4:
 pixel_values = pixel_values.unsqueeze(0) # Add batch dimension
 
 batch_size = pixel_values.shape[0]
 
 try:
 # FIXED: Process video with 32 frames
 outputs = self.videomae(pixel_values)
 
 # Get the sequence output (all tokens)
 sequence_output = outputs.last_hidden_state # Shape: (batch, seq_len, hidden_size)
 
 # Take the [CLS] token (first token) for classification
 cls_output = sequence_output[:, 0] # Shape: (batch, hidden_size)
 
 # Apply dropout
 pooled_output = self.dropout(cls_output)
 
 # Get final logits
 logits = self.classifier(pooled_output)
 
 return logits
 
 except Exception as e:
 print(f"Error in forward pass: {e}")
 # Return dummy output in case of error
 return torch.zeros(batch_size, len(EXERCISE_TYPES))

def train_epoch(model, dataloader, optimizer, criterion, device):
 """Train the model for one epoch"""
 model.train()
 running_loss = 0.0
 correct = 0
 total = 0
 successful_batches = 0
 
 progress_bar = tqdm(dataloader, desc="Training")
 for pixel_values, labels in progress_bar:
 pixel_values = pixel_values.to(device)
 labels = labels.to(device)
 
 optimizer.zero_grad()
 
 try:
 logits = model(pixel_values)
 loss = criterion(logits, labels)
 loss.backward()
 
 # Gradient clipping for stability
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
 optimizer.step()
 
 running_loss += loss.item()
 successful_batches += 1
 
 # Calculate accuracy (considering multi-label)
 predictions = torch.sigmoid(logits) > 0.5
 correct += (predictions == labels.bool()).all(dim=1).sum().item()
 total += labels.size(0)
 
 progress_bar.set_postfix({
 'loss': running_loss / successful_batches,
 'acc': 100 * correct / total if total > 0 else 0
 })
 
 except Exception as e:
 print(f"Error in training step: {e}")
 continue
 
 epoch_loss = running_loss / successful_batches if successful_batches > 0 else 0
 accuracy = 100 * correct / total if total > 0 else 0
 
 return epoch_loss, accuracy

def validate(model, dataloader, criterion, device):
 """Validate the model"""
 model.eval()
 running_loss = 0.0
 correct = 0
 total = 0
 successful_batches = 0
 
 all_preds = []
 all_labels = []
 all_probs = []
 
 with torch.no_grad():
 for frames, labels in tqdm(dataloader, desc="Validating"):
 frames = frames.to(device)
 labels = labels.to(device)
 
 try:
 logits = model(frames)
 loss = criterion(logits, labels)
 running_loss += loss.item()
 successful_batches += 1
 
 probabilities = torch.sigmoid(logits)
 predictions = probabilities > 0.5
 
 correct += (predictions == labels.bool()).all(dim=1).sum().item()
 total += labels.size(0)
 
 all_labels.append(labels.cpu().numpy())
 all_preds.append(predictions.cpu().numpy())
 all_probs.append(probabilities.cpu().numpy())
 
 except Exception as e:
 print(f"Error in validation step: {e}")
 continue
 
 epoch_loss = running_loss / successful_batches if successful_batches > 0 else 0
 accuracy = 100 * correct / total if total > 0 else 0
 
 if all_labels:
 all_labels = np.vstack(all_labels)
 all_preds = np.vstack(all_preds)
 all_probs = np.vstack(all_probs)
 else:
 # Return empty arrays if no successful batches
 all_labels = np.array([])
 all_preds = np.array([])
 all_probs = np.array([])
 
 return epoch_loss, accuracy, all_labels, all_preds, all_probs

def plot_confusion_matrices(y_true, y_pred, exercise_types, save_path=None):
 """Plot confusion matrices for each exercise type"""
 if len(y_true) == 0 or len(y_pred) == 0:
 print("No data available for confusion matrices")
 return None
 
 # Calculate multilabel confusion matrices
 cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)
 
 # Create figure with subplots
 n_exercises = len(exercise_types)
 n_cols = 4 # Number of columns in subplot grid
 n_rows = (n_exercises + n_cols - 1) // n_cols # Calculate required rows
 
 fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
 fig.suptitle('Confusion Matrices for Each Exercise Type', fontsize=16, fontweight='bold')
 
 # Flatten axes array for easier indexing
 if n_rows == 1:
 axes = axes.reshape(1, -1)
 axes_flat = axes.flatten()
 
 for i, exercise in enumerate(exercise_types):
 ax = axes_flat[i]
 
 # Get confusion matrix for this exercise
 cm = cm_multilabel[i]
 
 # Create labels for binary classification
 labels = ['Incorrect/Other', 'Correct']
 
 # Plot heatmap
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
 xticklabels=labels, yticklabels=labels, ax=ax,
 cbar_kws={'shrink': 0.8})
 
 ax.set_title(f'{exercise}', fontsize=12, fontweight='bold')
 ax.set_xlabel('Predicted', fontsize=10)
 ax.set_ylabel('Actual', fontsize=10)
 
 # Add performance metrics as text
 tn, fp, fn, tp = cm.ravel()
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0
 recall = tp / (tp + fn) if (tp + fn) > 0 else 0
 f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
 accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
 
 # Add text box with metrics
 textstr = f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}'
 props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
 ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
 verticalalignment='top', bbox=props)
 
 # Hide unused subplots
 for i in range(n_exercises, len(axes_flat)):
 axes_flat[i].set_visible(False)
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 print(f"Confusion matrices saved to: {save_path}")
 
 plt.show()
 
 return cm_multilabel

def plot_overall_confusion_matrix(y_true, y_pred, exercise_types, save_path=None):
 """Plot overall confusion matrix for multi-label classification"""
 if len(y_true) == 0 or len(y_pred) == 0:
 print("No data available for overall confusion matrix")
 return
 
 # Calculate per-sample accuracy
 sample_accuracy = (y_true == y_pred).all(axis=1)
 
 # Count correct vs incorrect predictions
 correct_count = np.sum(sample_accuracy)
 incorrect_count = len(sample_accuracy) - correct_count
 
 # Create simple 2x2 matrix
 overall_cm = np.array([[incorrect_count, 0], [0, correct_count]])
 
 plt.figure(figsize=(8, 6))
 sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=['Incorrect', 'Correct'],
 yticklabels=['Incorrect', 'Correct'],
 cbar_kws={'shrink': 0.8})
 
 plt.title('Overall Multi-Label Classification Results', fontsize=14, fontweight='bold')
 plt.xlabel('Predicted', fontsize=12)
 plt.ylabel('Actual', fontsize=12)
 
 # Add accuracy text
 total_samples = len(sample_accuracy)
 overall_accuracy = correct_count / total_samples
 plt.text(0.5, -0.1, f'Overall Exact Match Accuracy: {overall_accuracy:.3f} ({correct_count}/{total_samples})',
 ha='center', va='top', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 print(f"Overall confusion matrix saved to: {save_path}")
 
 plt.tight_layout()
 plt.show()

def plot_exercise_performance_summary(y_true, y_pred, exercise_types, save_path=None):
 """Plot summary of performance metrics for all exercises"""
 if len(y_true) == 0 or len(y_pred) == 0:
 print("No data available for performance summary")
 return pd.DataFrame()
 
 metrics = []
 
 for i, exercise in enumerate(exercise_types):
 y_true_exercise = y_true[:, i]
 y_pred_exercise = y_pred[:, i]
 
 if len(np.unique(y_true_exercise)) > 1:
 precision, recall, f1, _ = precision_recall_fscore_support(
 y_true_exercise, y_pred_exercise, average='binary', zero_division=0
 )
 accuracy = accuracy_score(y_true_exercise, y_pred_exercise)
 else:
 precision = recall = f1 = accuracy = 0.0
 
 metrics.append({
 'Exercise': exercise,
 'Accuracy': accuracy,
 'Precision': precision,
 'Recall': recall,
 'F1-Score': f1,
 'Support': np.sum(y_true_exercise)
 })
 
 metrics_df = pd.DataFrame(metrics)
 
 # Create subplot figure
 fig, axes = plt.subplots(2, 2, figsize=(16, 12))
 fig.suptitle('Performance Metrics Summary by Exercise', fontsize=16, fontweight='bold')
 
 # Plot 1: Accuracy
 axes[0,0].barh(metrics_df['Exercise'], metrics_df['Accuracy'], color='skyblue')
 axes[0,0].set_xlabel('Accuracy')
 axes[0,0].set_title('Accuracy by Exercise')
 axes[0,0].set_xlim(0, 1)
 
 # Plot 2: Precision
 axes[0,1].barh(metrics_df['Exercise'], metrics_df['Precision'], color='lightgreen')
 axes[0,1].set_xlabel('Precision')
 axes[0,1].set_title('Precision by Exercise')
 axes[0,1].set_xlim(0, 1)
 
 # Plot 3: Recall
 axes[1,0].barh(metrics_df['Exercise'], metrics_df['Recall'], color='lightcoral')
 axes[1,0].set_xlabel('Recall')
 axes[1,0].set_title('Recall by Exercise')
 axes[1,0].set_xlim(0, 1)
 
 # Plot 4: F1-Score
 axes[1,1].barh(metrics_df['Exercise'], metrics_df['F1-Score'], color='gold')
 axes[1,1].set_xlabel('F1-Score')
 axes[1,1].set_title('F1-Score by Exercise')
 axes[1,1].set_xlim(0, 1)
 
 # Rotate y-axis labels for better readability
 for ax in axes.flat:
 ax.tick_params(axis='y', labelsize=8)
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 print(f"Performance summary saved to: {save_path}")
 
 plt.show()
 
 return metrics_df

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
 train_accs, val_accs, best_val_loss, filename):
 """Save checkpoint for resuming training"""
 checkpoint = {
 'epoch': epoch,
 'model_state_dict': model.state_dict(),
 'optimizer_state_dict': optimizer.state_dict(),
 'scheduler_state_dict': scheduler.state_dict(),
 'train_losses': train_losses,
 'val_losses': val_losses,
 'train_accs': train_accs,
 'val_accs': val_accs,
 'best_val_loss': best_val_loss,
 'model_type': 'VideoMAE_32frames', # Updated identifier
 'num_exercises': NUM_EXERCISES,
 'exercise_types': EXERCISE_TYPES,
 'num_frames': NUM_FRAMES
 }
 torch.save(checkpoint, filename)
 print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
 """Load checkpoint to resume training"""
 if os.path.isfile(filename):
 print(f"Loading checkpoint: {filename}")
 checkpoint = torch.load(filename, map_location=DEVICE)
 
 # Check compatibility
 if 'model_type' not in checkpoint or 'VideoMAE' not in checkpoint['model_type']:
 print(f"Warning: Checkpoint appears to be from a different model type.")
 print("Starting training from scratch...")
 return 0, [], [], [], [], float('inf')
 
 # Check if exercise types match
 if 'exercise_types' in checkpoint:
 saved_exercises = checkpoint['exercise_types']
 if saved_exercises != EXERCISE_TYPES:
 print(f"Warning: Exercise types have changed!")
 print(f"Saved: {saved_exercises}")
 print(f"Current: {EXERCISE_TYPES}")
 print("Starting training from scratch...")
 return 0, [], [], [], [], float('inf')
 
 # Check if number of frames matches
 if 'num_frames' in checkpoint:
 saved_frames = checkpoint['num_frames']
 if saved_frames != NUM_FRAMES:
 print(f"Warning: Number of frames has changed from {saved_frames} to {NUM_FRAMES}!")
 print("Starting training from scratch...")
 return 0, [], [], [], [], float('inf')
 
 try:
 model.load_state_dict(checkpoint['model_state_dict'])
 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
 
 start_epoch = checkpoint['epoch'] + 1
 train_losses = checkpoint['train_losses']
 val_losses = checkpoint['val_losses']
 train_accs = checkpoint['train_accs']
 val_accs = checkpoint['val_accs']
 best_val_loss = checkpoint['best_val_loss']
 
 print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
 return start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_loss
 except Exception as e:
 print(f"Error loading checkpoint: {e}")
 print("Starting training from scratch...")
 return 0, [], [], [], [], float('inf')
 else:
 print(f"No checkpoint found at {filename}")
 return 0, [], [], [], [], float('inf')

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
 """Plot training and validation metrics"""
 plt.figure(figsize=(12, 5))
 
 plt.subplot(1, 2, 1)
 plt.plot(train_losses, label='Train Loss')
 plt.plot(val_losses, label='Validation Loss')
 plt.title('Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.legend()
 plt.grid(True, alpha=0.3)
 
 plt.subplot(1, 2, 2)
 plt.plot(train_accs, label='Train Accuracy')
 plt.plot(val_accs, label='Validation Accuracy')
 plt.title('Accuracy')
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy (%)')
 plt.legend()
 plt.grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.savefig('videomae_multilabel_training_history.png', dpi=300)
 plt.show()

def evaluate_model(model, test_dataloader, device):
 """Evaluate the model on test data"""
 model.eval()
 
 all_preds = []
 all_labels = []
 all_probs = []
 
 with torch.no_grad():
 for frames, labels in tqdm(test_dataloader, desc="Testing"):
 frames = frames.to(device)
 
 try:
 logits = model(frames)
 probabilities = torch.sigmoid(logits)
 predictions = probabilities > 0.5
 
 all_preds.append(predictions.cpu().numpy())
 all_labels.append(labels.numpy())
 all_probs.append(probabilities.cpu().numpy())
 
 except Exception as e:
 print(f"Error in evaluation: {e}")
 continue
 
 if all_preds:
 all_preds = np.vstack(all_preds)
 all_labels = np.vstack(all_labels)
 all_probs = np.vstack(all_probs)
 
 # Calculate per-exercise metrics
 print("\n" + "="*80)
 print("PER-EXERCISE CLASSIFICATION RESULTS")
 print("="*80)
 
 for i, exercise in enumerate(EXERCISE_TYPES):
 exercise_labels = all_labels[:, i]
 exercise_preds = all_preds[:, i]
 
 if len(np.unique(exercise_labels)) > 1: # Only if both classes present
 accuracy = accuracy_score(exercise_labels, exercise_preds)
 precision, recall, f1, _ = precision_recall_fscore_support(
 exercise_labels, exercise_preds, average='binary', zero_division=0
 )
 
 print(f"\n{exercise}:")
 print(f" Accuracy: {accuracy:.4f}")
 print(f" Precision: {precision:.4f}")
 print(f" Recall: {recall:.4f}")
 print(f" F1-Score: {f1:.4f}")
 print(f" Samples: {len(exercise_labels)} (Positive: {exercise_labels.sum():.0f})")
 else:
 print("No successful predictions made during evaluation.")
 all_preds = np.array([])
 all_labels = np.array([])
 all_probs = np.array([])
 
 return all_preds, all_labels, all_probs

def predict_single_video(model, video_path, device):
 """Make prediction on a single video"""
 model.eval()
 
 frames = extract_frames(video_path)
 
 try:
 # Process frames manually (same as in dataset)
 frames_array = np.stack(frames)
 frames_tensor = torch.tensor(frames_array, dtype=torch.float32) / 255.0
 frames_tensor = frames_tensor.permute(0, 3, 1, 2) # (32, 3, 224, 224)
 frames_tensor = frames_tensor.unsqueeze(0).to(device) # Add batch dimension
 
 with torch.no_grad():
 logits = model(frames_tensor)
 probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
 predictions = probabilities > 0.5
 
 print(f"\nVideo: {os.path.basename(video_path)}")
 print(f"Predictions:")
 
 for i, exercise in enumerate(EXERCISE_TYPES):
 status = "CORRECT" if predictions[i] else "INCORRECT"
 confidence = probabilities[i] if predictions[i] else (1 - probabilities[i])
 print(f" {exercise}: {status} (Confidence: {confidence*100:.2f}%)")
 
 return predictions, probabilities
 
 except Exception as e:
 print(f"Error predicting video {video_path}: {e}")
 return np.zeros(NUM_EXERCISES), np.ones(NUM_EXERCISES) * 0.5

def cleanup_incompatible_checkpoints():
 """Remove incompatible checkpoints"""
 checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
 if os.path.exists(checkpoint_path):
 try:
 checkpoint = torch.load(checkpoint_path, map_location='cpu')
 if ('model_type' not in checkpoint or 
 'VideoMAE' not in checkpoint['model_type'] or
 checkpoint.get('num_frames', 16) != NUM_FRAMES):
 print("Removing incompatible checkpoint...")
 os.remove(checkpoint_path)
 print("Incompatible checkpoint removed.")
 except Exception as e:
 print(f"Error checking checkpoint compatibility: {e}")
 print("Removing potentially corrupted checkpoint...")
 os.remove(checkpoint_path)

def main():
 print(f"PyTorch version: {torch.__version__}")
 print(f"Transformers version: {transformers.__version__}")
 
 # Clean up incompatible checkpoints
 cleanup_incompatible_checkpoints()
 
 print("Loading video data...")
 df = load_video_data()
 
 if len(df) == 0:
 print("Error: No videos found! Please check your dataset path.")
 return
 
 print(f"Total videos: {len(df)}")
 print(f"Exercise distribution:")
 print(df['exercise_type'].value_counts())
 print(f"Correctness distribution:")
 print(df['correctness'].value_counts())
 
 # Create multi-label targets (this will also update EXERCISE_TYPES)
 print("Creating multi-label targets...")
 multi_labels = create_multi_label_targets(df)
 print(f"Multi-label shape: {multi_labels.shape}")
 
 # Split data
 train_df, temp_df, train_labels, temp_labels = train_test_split(
 df, multi_labels, test_size=0.3, random_state=42
 )
 val_df, test_df, val_labels, test_labels = train_test_split(
 temp_df, temp_labels, test_size=0.5, random_state=42
 )
 
 print(f"Training videos: {len(train_df)}")
 print(f"Validation videos: {len(val_df)}")
 print(f"Testing videos: {len(test_df)}")
 
 # Create datasets (no need for processor now)
 train_dataset = PhysioVideoDataset(train_df, train_labels, None, is_training=True)
 val_dataset = PhysioVideoDataset(val_df, val_labels, None, is_training=False)
 test_dataset = PhysioVideoDataset(test_df, test_labels, None, is_training=False)
 
 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
 test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
 
 print("Initializing VideoMAE model for multi-label classification with 32 frames...")
 model = VideoMAEForMultiLabelClassification(num_exercises=NUM_EXERCISES)
 model = model.to(DEVICE)
 
 # Multi-label loss function
 criterion = nn.BCEWithLogitsLoss()
 
 optimizer = optim.AdamW(
 filter(lambda p: p.requires_grad, model.parameters()),
 lr=LEARNING_RATE,
 weight_decay=1e-3
 )
 
 scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 optimizer, mode='min', factor=0.1, patience=3
 )
 
 start_epoch = 0
 train_losses = []
 val_losses = []
 train_accs = []
 val_accs = []
 best_val_loss = float('inf')
 
 last_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
 
 if RESUME_TRAINING and os.path.exists(last_checkpoint_path):
 start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_loss = load_checkpoint(
 model, optimizer, scheduler, last_checkpoint_path
 )
 
 # Training loop
 patience = 7
 early_stop_counter = 0
 
 print("Training VideoMAE model for multi-label classification...")
 for epoch in range(start_epoch, EPOCHS):
 print(f"\nEpoch {epoch+1}/{EPOCHS}")
 
 train_loss, train_acc = train_epoch(
 model, train_dataloader, optimizer, criterion, DEVICE
 )
 
 val_loss, val_acc, y_true, y_pred, y_probs = validate(
 model, val_dataloader, criterion, DEVICE
 )
 
 scheduler.step(val_loss)
 
 print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
 print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
 
 train_losses.append(train_loss)
 val_losses.append(val_loss)
 train_accs.append(train_acc)
 val_accs.append(val_acc)
 
 # Save checkpoints
 save_checkpoint(
 epoch, model, optimizer, scheduler,
 train_losses, val_losses, train_accs, val_accs, best_val_loss,
 last_checkpoint_path
 )
 
 if val_loss < best_val_loss:
 best_val_loss = val_loss
 early_stop_counter = 0
 print("Saving best model...")
 torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_videomae_model.pth'))
 else:
 early_stop_counter += 1
 
 # Early stopping
 if early_stop_counter >= patience:
 print(f"Early stopping triggered after {epoch+1} epochs")
 break
 
 # Plot training history
 if train_losses and val_losses:
 plot_training_history(train_losses, val_losses, train_accs, val_accs)
 
 # Evaluate on test set
 print("\nEvaluating on test set...")
 test_preds, test_labels, test_probs = evaluate_model(model, test_dataloader, DEVICE)
 
 if len(test_preds) > 0:
 # Generate and save confusion matrices
 print("\n" + "="*80)
 print("GENERATING CONFUSION MATRICES")
 print("="*80)
 
 # Plot individual confusion matrices for each exercise
 cm_multilabel = plot_confusion_matrices(
 test_labels, test_preds, EXERCISE_TYPES, 
 save_path=os.path.join(CHECKPOINT_DIR, 'confusion_matrices_individual.png')
 )
 
 # Plot overall confusion matrix
 plot_overall_confusion_matrix(
 test_labels, test_preds, EXERCISE_TYPES,
 save_path=os.path.join(CHECKPOINT_DIR, 'confusion_matrix_overall.png')
 )
 
 # Plot performance summary
 metrics_df = plot_exercise_performance_summary(
 test_labels, test_preds, EXERCISE_TYPES,
 save_path=os.path.join(CHECKPOINT_DIR, 'performance_summary.png')
 )
 
 # Save metrics to CSV
 if not metrics_df.empty:
 metrics_df.to_csv(os.path.join(CHECKPOINT_DIR, 'performance_metrics.csv'), index=False)
 
 # Calculate and print overall metrics
 print("\n" + "="*80)
 print("OVERALL TEST SET METRICS")
 print("="*80)
 print("\nClassification Report:")
 print(classification_report(test_labels, test_preds, target_names=EXERCISE_TYPES, zero_division=0))
 
 # Calculate overall exact match accuracy
 exact_match_accuracy = np.mean((test_labels == test_preds).all(axis=1))
 print(f"\nExact Match Accuracy: {exact_match_accuracy:.4f}")
 
 # Calculate Hamming loss (average per-label classification error)
 hamming_loss = np.mean(test_labels != test_preds)
 print(f"Hamming Loss: {hamming_loss:.4f}")
 
 # Example prediction on a single video
 print("\nTesting prediction on a single video...")
 sample_video = test_df['video_path'].iloc[0]
 predictions, probabilities = predict_single_video(model, sample_video, DEVICE)
 
 # Save training configuration and results
 results = {
 'exercise_types': EXERCISE_TYPES,
 'num_exercises': NUM_EXERCISES,
 'num_frames': NUM_FRAMES,
 'final_train_loss': train_losses[-1] if train_losses else None,
 'final_val_loss': val_losses[-1] if val_losses else None,
 'final_train_acc': train_accs[-1] if train_accs else None,
 'final_val_acc': val_accs[-1] if val_accs else None,
 'test_metrics': {
 'exact_match_accuracy': float(exact_match_accuracy),
 'hamming_loss': float(hamming_loss),
 'classification_report': classification_report(test_labels, test_preds, 
 target_names=EXERCISE_TYPES, 
 output_dict=True, 
 zero_division=0)
 }
 }
 
 with open(os.path.join(CHECKPOINT_DIR, 'training_results.json'), 'w') as f:
 json.dump(results, f, indent=2)
 
 # Save final model
 print("\nSaving final model...")
 torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_videomae_model.pth'))
 
 print("\nTraining completed!")
 print(f"Results saved to {CHECKPOINT_DIR}/training_results.json")
 print(f"Performance metrics saved to {CHECKPOINT_DIR}/performance_metrics.csv")
 print(f"Confusion matrices saved to {CHECKPOINT_DIR}/")
 print(f"Best model saved to {CHECKPOINT_DIR}/best_videomae_model.pth")
 print(f"Final model saved to {CHECKPOINT_DIR}/final_videomae_model.pth")

if __name__ == "__main__":
 main()
