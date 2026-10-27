from src.utils import analyze_dataset
from src.approach1_classical import train_classical_model
from src.approach2_dl import train_dl_model


DATASET_DIR = "./train" 

def main():
    print("Step 1: Running Exploratory Data Analysis (EDA)...")
    eda_df = analyze_dataset(DATASET_DIR)
    if eda_df is None:
        return
    
    print("\nStep 2: Training Approach 1 (Classical Color Histograms + SVM)...")
    _, class_names, _, _ = train_classical_model(DATASET_DIR)
    
    print("\nStep 3: Training Approach 2 (Lightweight Deep Learning - MobileNetV2)...")
    train_dl_model(DATASET_DIR, epochs=3)
    
    print("\n🎉 Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()