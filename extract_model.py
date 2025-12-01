import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from models import VictimCNN, StolenCNN

def knowledge_distillation_loss(student_logits, teacher_probs, temperature=3.0):
    """
    Soft target loss for knowledge distillation.
    Higher temperature produces softer probability distributions.
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(soft_student, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)


def extract_model(victim_model, num_queries=1000, epochs=20, batch_size=64):
    """
    Perform model extraction attack via knowledge distillation.

    The attacker:
    1. Has NO access to victim's training data or architecture
    2. Can only query the victim model (black-box access)
    3. Uses victim's soft predictions to train a substitute model

    Args:
        victim_model: The target model to steal
        num_queries: Number of queries to the victim (attack budget)
        epochs: Training epochs for stolen model
        batch_size: Batch size for training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting model using {device}")
    print(f"Attack budget: {num_queries} queries to victim model")

    victim_model.eval()
    victim_model.to(device)

    # Load a DIFFERENT dataset for querying (simulating attacker's data)
    # In practice, attacker might use public data or synthetic images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Use test set as "attacker's query dataset" (attacker doesn't have training data)
    query_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Limit to num_queries (simulating API rate limits / attack budget)
    indices = np.random.choice(len(query_dataset), num_queries, replace=False)
    query_subset = Subset(query_dataset, indices)
    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=True)

    # Step 1: Query victim model to collect soft labels
    print("\n[Phase 1] Querying victim model...")
    query_images = []
    soft_labels = []

    with torch.no_grad():
        for images, _ in query_loader:  # Ignore true labels - attacker doesn't have them!
            images = images.to(device)
            probs = victim_model.predict_proba(images)
            query_images.append(images.cpu())
            soft_labels.append(probs.cpu())

    query_images = torch.cat(query_images)
    soft_labels = torch.cat(soft_labels)
    print(f"Collected {len(query_images)} query-response pairs")

    # Step 2: Train stolen model on soft labels
    print("\n[Phase 2] Training stolen model via knowledge distillation...")
    stolen_model = StolenCNN().to(device)
    optimizer = optim.Adam(stolen_model.parameters(), lr=0.001)

    # Create dataset from collected queries
    extraction_dataset = torch.utils.data.TensorDataset(query_images, soft_labels)
    extraction_loader = DataLoader(extraction_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        stolen_model.train()
        total_loss = 0

        for images, teacher_probs in extraction_loader:
            images = images.to(device)
            teacher_probs = teacher_probs.to(device)

            optimizer.zero_grad()
            student_logits = stolen_model(images)
            loss = knowledge_distillation_loss(student_logits, teacher_probs, temperature=3.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Distillation Loss: {total_loss/len(extraction_loader):.4f}')

    # Step 3: Evaluate extraction success
    print("\n[Phase 3] Evaluating extraction success...")
    stolen_model.eval()

    # Test on held-out data
    test_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    victim_preds = []
    stolen_preds = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            victim_out = victim_model(images)
            stolen_out = stolen_model(images)

            victim_preds.extend(victim_out.argmax(1).cpu().numpy())
            stolen_preds.extend(stolen_out.argmax(1).cpu().numpy())
            true_labels.extend(labels.numpy())

    victim_preds = np.array(victim_preds)
    stolen_preds = np.array(stolen_preds)
    true_labels = np.array(true_labels)

    # Agreement rate: How often stolen model matches victim
    agreement = (victim_preds == stolen_preds).mean() * 100

    # Accuracy metrics
    victim_acc = (victim_preds == true_labels).mean() * 100
    stolen_acc = (stolen_preds == true_labels).mean() * 100

    print(f"\n{'='*50}")
    print(f"EXTRACTION RESULTS")
    print(f"{'='*50}")
    print(f"Victim Model Accuracy:  {victim_acc:.2f}%")
    print(f"Stolen Model Accuracy:  {stolen_acc:.2f}%")
    print(f"Agreement Rate:         {agreement:.2f}%")
    print(f"Fidelity (stolen/victim): {stolen_acc/victim_acc*100:.1f}%")
    print(f"{'='*50}")

    # Save stolen model
    torch.save(stolen_model.state_dict(), 'stolen_model.pt')
    print("\nStolen model saved to 'stolen_model.pt'")

    return stolen_model, {
        'agreement': agreement,
        'victim_acc': victim_acc,
        'stolen_acc': stolen_acc,
        'num_queries': num_queries
    }


if __name__ == '__main__':
    # Load pre-trained victim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    victim = VictimCNN().to(device)
    victim.load_state_dict(torch.load('victim_model.pt', map_location=device))

    # Perform extraction with limited queries
    extract_model(victim, num_queries=1000, epochs=20)
