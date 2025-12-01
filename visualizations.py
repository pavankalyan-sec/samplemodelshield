import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from models import VictimCNN, StolenCNN, CLASS_NAMES

def load_models():
    """Load both trained models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    victim = VictimCNN().to(device)
    victim.load_state_dict(torch.load('victim_model.pt', map_location=device))
    victim.eval()

    stolen = StolenCNN().to(device)
    stolen.load_state_dict(torch.load('stolen_model.pt', map_location=device))
    stolen.eval()

    return victim, stolen, device


def generate_comparison_grid(num_samples=8, save_path='static/comparison_grid.png'):
    """
    Generate side-by-side comparison grid showing:
    - Original image
    - Victim model prediction + confidence
    - Stolen model prediction + confidence
    - Agreement indicator
    """
    victim, stolen, device = load_models()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Select diverse samples
    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig = plt.figure(figsize=(16, num_samples * 2))
    gs = GridSpec(num_samples, 4, figure=fig, width_ratios=[1, 1.5, 1.5, 0.5])

    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            victim_probs = F.softmax(victim(image_tensor), dim=1)[0]
            stolen_probs = F.softmax(stolen(image_tensor), dim=1)[0]

        victim_pred = victim_probs.argmax().item()
        stolen_pred = stolen_probs.argmax().item()
        victim_conf = victim_probs[victim_pred].item() * 100
        stolen_conf = stolen_probs[stolen_pred].item() * 100

        # Denormalize image for display
        img_display = image.squeeze().numpy() * 0.5 + 0.5

        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(img_display, cmap='gray')
        ax1.set_title(f'True: {CLASS_NAMES[true_label]}', fontsize=10)
        ax1.axis('off')

        # Victim prediction bar
        ax2 = fig.add_subplot(gs[i, 1])
        colors = ['#2ecc71' if j == victim_pred else '#bdc3c7' for j in range(10)]
        bars = ax2.barh(range(10), victim_probs.cpu().numpy(), color=colors)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([c[:8] for c in CLASS_NAMES], fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_title(f'Victim: {CLASS_NAMES[victim_pred]} ({victim_conf:.1f}%)', fontsize=10)
        ax2.invert_yaxis()

        # Stolen prediction bar
        ax3 = fig.add_subplot(gs[i, 2])
        colors = ['#e74c3c' if j == stolen_pred else '#bdc3c7' for j in range(10)]
        ax3.barh(range(10), stolen_probs.cpu().numpy(), color=colors)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([c[:8] for c in CLASS_NAMES], fontsize=8)
        ax3.set_xlim(0, 1)
        ax3.set_title(f'Stolen: {CLASS_NAMES[stolen_pred]} ({stolen_conf:.1f}%)', fontsize=10)
        ax3.invert_yaxis()

        # Agreement indicator
        ax4 = fig.add_subplot(gs[i, 3])
        if victim_pred == stolen_pred:
            ax4.text(0.5, 0.5, 'MATCH', fontsize=12, ha='center', va='center', color='#2ecc71', fontweight='bold')
            ax4.set_facecolor('#d5f5e3')
        else:
            ax4.text(0.5, 0.5, 'DIFF', fontsize=12, ha='center', va='center', color='#e74c3c', fontweight='bold')
            ax4.set_facecolor('#fadbd8')
        ax4.axis('off')

    plt.suptitle('Model Extraction Attack: Victim vs Stolen Model Predictions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison grid to {save_path}")


def generate_confusion_matrices(save_path='static/confusion_matrices.png'):
    """Generate side-by-side confusion matrices for victim and stolen models."""
    victim, stolen, device = load_models()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    all_labels = []
    victim_preds = []
    stolen_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            victim_preds.extend(victim(images).argmax(1).cpu().numpy())
            stolen_preds.extend(stolen(images).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    # Create confusion matrices
    victim_cm = confusion_matrix(all_labels, victim_preds)
    stolen_cm = confusion_matrix(all_labels, stolen_preds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Victim confusion matrix
    sns.heatmap(victim_cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=[c[:6] for c in CLASS_NAMES],
                yticklabels=[c[:6] for c in CLASS_NAMES])
    axes[0].set_title('Victim Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True Label')

    # Stolen confusion matrix
    sns.heatmap(stolen_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=[c[:6] for c in CLASS_NAMES],
                yticklabels=[c[:6] for c in CLASS_NAMES])
    axes[1].set_title('Stolen Model', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True Label')

    plt.suptitle('Confusion Matrix Comparison: Similar Error Patterns Indicate Successful Extraction',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved confusion matrices to {save_path}")


def generate_agreement_chart(save_path='static/agreement_chart.png'):
    """Generate per-class agreement rate chart."""
    victim, stolen, device = load_models()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    class_agreements = {i: [] for i in range(10)}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            victim_pred = victim(images).argmax(1).cpu().numpy()
            stolen_pred = stolen(images).argmax(1).cpu().numpy()

            for v, s, label in zip(victim_pred, stolen_pred, labels.numpy()):
                class_agreements[label].append(v == s)

    agreement_rates = [np.mean(class_agreements[i]) * 100 for i in range(10)]
    overall_agreement = np.mean([a for agreements in class_agreements.values() for a in agreements]) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db' if rate >= 90 else '#f39c12' if rate >= 80 else '#e74c3c'
              for rate in agreement_rates]

    bars = ax.bar(CLASS_NAMES, agreement_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=overall_agreement, color='#2c3e50', linestyle='--', linewidth=2,
               label=f'Overall: {overall_agreement:.1f}%')

    # Add value labels on bars
    for bar, rate in zip(bars, agreement_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_xlabel('Fashion Category', fontsize=12)
    ax.set_title('Stolen Model Agreement with Victim Model by Category\n'
                 '(Higher = More Successful Extraction)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=11)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved agreement chart to {save_path}")


def generate_all_visualizations():
    """Generate all static visualizations for the blog post."""
    import os
    os.makedirs('static', exist_ok=True)

    print("Generating comparison grid...")
    generate_comparison_grid()

    print("Generating confusion matrices...")
    generate_confusion_matrices()

    print("Generating agreement chart...")
    generate_agreement_chart()

    print("\nAll visualizations generated in static/ directory")


if __name__ == '__main__':
    generate_all_visualizations()
