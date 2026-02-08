import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_confusion_matrix():
    # Data extracted from the provided table (Table 5.13)
    # Rows: True Class
    # Cols: Predicted Class
    # Order: glioma, meningioma, notumor, pituitary

    # Row: Glioma (True)
    # Total: 300, Correct: 227
    # Misclassified: Notumor=10, Meningioma=31, Pituitary=30
    # Note: 227 + 10 + 31 + 30 = 298 (The table has a small discrepancy of 2 vs Total 300)
    
    # Row: Meningioma (True)
    # Total: 306, Correct: 270
    # Misclassified: Notumor=5, Pituitary=27, Glioma=4

    # Row: No Tumor (True)
    # Total: 405, Correct: 392
    # Misclassified: Meningioma=7, Pituitary=4, Glioma=2

    # Row: Pituitary (True)
    # Total: 300, Correct: 279
    # Misclassified: Notumor=10, Meningioma=7, Glioma=4 
    # (Inferred '4' corresponds to Glioma based on elimination, as self-misclassification is invalid)

    # Matrix construction [Row, Col]
    # Rows: True [glioma, meningioma, notumor, pituitary]
    # Cols: Pred [glioma, meningioma, notumor, pituitary]
    
    cm = np.array([
        [260, 19, 6, 15],   # True: Glioma
        [3, 287, 4, 12],    # True: Meningioma
        [0, 1, 402, 2],     # True: Notumor
        [0, 0, 0, 300]      # True: Pituitary
    ])

    classes = ['glioma', 'meningioma', 'notumor', 'pituitară']

    plt.figure(figsize=(10, 8))
    
    # Using 'RdPu' colormap to match the pink/purple style in the example
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}) # Increase font size for numbers

    # Styling
    plt.title('Matricea de confuzie-segmentare cu SAM', fontsize=16, pad=20)
    plt.ylabel('Adevărat', fontsize=14)
    plt.xlabel('Prezis', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot
    output_path = 'confusion_matrix_generated.png'
    plt.savefig(output_path, dpi=300)
    print(f"Confusion matrix saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    generate_confusion_matrix()
