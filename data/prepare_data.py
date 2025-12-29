import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def load_classes(classes_csv):
    """Load class mapping from CSV (format: class_name,class_id with no header)"""
    class_map = {}
    with open(classes_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                class_name, class_id = line.split(',')
                class_map[int(class_id)] = class_name
    return class_map


# prepare labels in required csv format with img_path, x1, y1, x2, y2, class_name
# YOLO format: class_id, x_center, y_center, width, height (all normalized)
def prepare_labels(image_paths, labels_dir, class_map):
    csv_labels = []
    skipped = 0
    for img_path in tqdm(image_paths):
        W, H = Image.open(img_path).size
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'r') as f:
            labels = f.readlines()
        for label in labels:
            class_id, x_center, y_center, bw, bh = map(float, label.strip().split())
            x1 = int((x_center - bw / 2) * W)
            y1 = int((y_center - bh / 2) * H)
            x2 = int((x_center + bw / 2) * W)
            y2 = int((y_center + bh / 2) * H)
            
            # Ensure minimum box size of 1 pixel
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            # Skip if still invalid after clamping
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
                
            class_name = class_map[int(class_id)]
            csv_labels.append([img_path, x1, y1, x2, y2, class_name])
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} invalid bounding boxes")
    
    df = pd.DataFrame(csv_labels, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'class_name'])
    return df


def test_label(df, output_path="test_bbox.png"):
    for index, row in df.iterrows():
        img_path = row['img_path']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        img = Image.open(img_path)
        plt.figure()
        plt.imshow(img)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2))
        plt.savefig(output_path)
        plt.close()
        input("Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO labels to CSV format')
    parser.add_argument('--dataset-dir', type=str, 
                        default='/home/anudeep/.cache/kagglehub/datasets/fareselmenshawii/face-detection-dataset/versions/3',
                        help='Path to dataset directory')
    parser.add_argument('--classes-csv', type=str, required=True,
                        help='Path to classes.csv (format: class_name,class_id)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output CSV files')
    parser.add_argument('--test', action='store_true',
                        help='Run visual test on a few samples')
    parser.add_argument('--num-test', type=int, default=5,
                        help='Number of samples to test (with --test)')
    args = parser.parse_args()

    # Load class mapping
    class_map = load_classes(args.classes_csv)
    print(f"Loaded classes: {class_map}")

    train_images_dir = os.path.join(args.dataset_dir, 'images/train')
    train_labels_dir = os.path.join(args.dataset_dir, 'labels/train')
    val_images_dir = os.path.join(args.dataset_dir, 'images/val')
    val_labels_dir = os.path.join(args.dataset_dir, 'labels/val')

    train_image_paths = [os.path.join(train_images_dir, fname) for fname in os.listdir(train_images_dir) if fname.endswith('.jpg')]
    val_image_paths = [os.path.join(val_images_dir, fname) for fname in os.listdir(val_images_dir) if fname.endswith('.jpg')]
    
    print(f"Number of training images: {len(train_image_paths)}")
    print(f"Number of validation images: {len(val_image_paths)}")

    train_df = prepare_labels(train_image_paths, train_labels_dir, class_map)
    val_df = prepare_labels(val_image_paths, val_labels_dir, class_map)

    os.makedirs(args.output_dir, exist_ok=True)
    train_csv = os.path.join(args.output_dir, 'train_labels.csv')
    val_csv = os.path.join(args.output_dir, 'val_labels.csv')
    
    # Save without header for pytorch-retinanet compatibility
    train_df.to_csv(train_csv, index=False, header=False)
    val_df.to_csv(val_csv, index=False, header=False)
    print(f"Saved: {train_csv}, {val_csv}")

    print("Sample training labels:")
    print(train_df.head())
    print("Sample validation labels:")
    print(val_df.head())

    if args.test:
        print(f"\nRunning visual test on {args.num_test} samples...")
        test_label(train_df.head(args.num_test))


if __name__ == "__main__":
    main()