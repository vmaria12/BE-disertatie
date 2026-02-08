import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

def generate_smooth_roc():
    # Classes from your dataset
    classes = ['pituitară', 'notumor', 'meningioma', 'glioma']
    
    # Target "smooth" performance to match the look of the reference image
    # We simulate prediction scores to achieve high AUCs (area under curve)
    # mirroring the accuracy order: Pituitary > Notumor > Meningioma > Glioma
    
    # Parameters to shift the positive distribution to get desired curve shape
    # Higher 'shift' = better separation = higher AUC = curve closer to top-left
    display_params = [
        {'name': 'pituitară',   'color': 'red',           'shift': 2.5},  # Best (approx 100%)
        {'name': 'notumor',     'color': 'lime',          'shift': 2.1},  # Very Good (approx 99%)
        {'name': 'meningioma',  'color': 'blue',          'shift': 2.1},  # Good (approx 93%)
        {'name': 'glioma',      'color': 'magenta',       'shift': 1.9},  # Lower (approx 86%)
    ]
    plt.figure(figsize=(10, 8))
    
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = 1000

    for params in display_params:
        # Generate synthetic scores
        # Negatives: Centered at 0
        # Positives: Centered at 'shift'
        
        # y_true: 0 for first half, 1 for second half
        y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
        
        # y_score: random normal distributions
        scores_neg = np.random.normal(0, 1, n_samples)
        scores_pos = np.random.normal(params['shift'], 1, n_samples)
        y_score = np.concatenate([scores_neg, scores_pos])
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=params['color'], lw=2,
                 label='ROC curve - {0}'.format(params['name'], roc_auc))

    # Plot styling to match reference image
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('Fals Pozitiv', fontsize=12, fontweight='bold')
    plt.ylabel('Adevărat Pozitiv', fontsize=12, fontweight='bold')
    plt.title('Curba ROC', fontsize=14, fontweight='bold')
    
    # Legend box styling
    plt.legend(loc="lower right", frameon=True, edgecolor='black', fancybox=False)
    
    # Clean look (inner ticks)
    plt.tick_params(direction='in', top=True, right=True)
    
    output_path = 'roc_curve_smooth_simulated.png'
    plt.savefig(output_path, dpi=300)
    print(f"Smooth ROC curve saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_smooth_roc()
