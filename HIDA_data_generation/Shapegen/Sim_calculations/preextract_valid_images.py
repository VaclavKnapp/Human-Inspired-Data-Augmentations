import os
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

def load_dino_v2_model():
    """Load DINO v2 model"""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

def extract_feature(model, image_path, transform, device):
    """Extract feature vector from a single image"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(input_tensor)
    return feature.cpu().squeeze().numpy()

def extract_features_for_unique_images(unique_images_file, output_path):
    """Extract features only for the unique images from precomputed pairs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load unique images list
    with open(unique_images_file, 'rb') as f:
        unique_images = pickle.load(f)
    
    print(f"Loaded {len(unique_images)} unique images to process")
    
    # Create output directory
    features_dir = os.path.join(output_path, 'extracted_features_pairs')
    os.makedirs(features_dir, exist_ok=True)
    
    # Load model
    print("Loading DINO v2 model...")
    model = load_dino_v2_model().to(device)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract features with progress bar
    print("Extracting features...")
    features_dict = {}
    
    for img_path in tqdm(unique_images, desc="Extracting features"):
        try:
            feature = extract_feature(model, img_path, transform, device)
            features_dict[img_path] = feature
        except Exception as e:
            print(f"\nError extracting feature from {img_path}: {e}")
            continue
    
    # Save all features in a single file for easy loading
    features_file = os.path.join(features_dir, 'all_features.pkl')
    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\nExtraction complete! Features saved to: {features_file}")
    print(f"Successfully extracted features for {len(features_dict)}/{len(unique_images)} images")
    
    # Also save as numpy arrays for memory efficiency if needed
    features_array = np.array(list(features_dict.values()))
    image_paths_array = list(features_dict.keys())
    
    npz_file = os.path.join(features_dir, 'all_features.npz')
    np.savez_compressed(npz_file,
                       features=features_array,
                       image_paths=image_paths_array)
    print(f"Also saved as compressed numpy array: {npz_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features for precomputed unique images')
    parser.add_argument('--unique_images_file', type=str, required=True,
                        help='Path to the unique_images.pkl file from precomputation')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for extracted features')
    
    args = parser.parse_args()
    
    extract_features_for_unique_images(args.unique_images_file, args.output_path)