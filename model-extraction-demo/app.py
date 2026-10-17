import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from models import VictimCNN, StolenCNN, CLASS_NAMES

# Global model instances
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
victim_model = None
stolen_model = None
test_dataset = None


def load_resources():
    """Load models and dataset on startup."""
    global victim_model, stolen_model, test_dataset

    victim_model = VictimCNN().to(device)
    victim_model.load_state_dict(torch.load('victim_model.pt', map_location=device))
    victim_model.eval()

    stolen_model = StolenCNN().to(device)
    stolen_model.load_state_dict(torch.load('stolen_model.pt', map_location=device))
    stolen_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )


def get_random_sample():
    """Get a random sample from the test dataset."""
    idx = np.random.randint(len(test_dataset))
    image, label = test_dataset[idx]

    # Convert to displayable format
    img_display = (image.squeeze().numpy() * 0.5 + 0.5) * 255
    img_pil = Image.fromarray(img_display.astype(np.uint8), mode='L')

    return img_pil, CLASS_NAMES[label], idx


def predict_and_compare(image):
    """Run both models on input and return comparison."""
    if image is None:
        return None, "No image provided", "", ""

    # Preprocess
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        victim_logits = victim_model(image_tensor)
        stolen_logits = stolen_model(image_tensor)

        victim_probs = F.softmax(victim_logits, dim=1)[0].cpu().numpy()
        stolen_probs = F.softmax(stolen_logits, dim=1)[0].cpu().numpy()

    victim_pred = np.argmax(victim_probs)
    stolen_pred = np.argmax(stolen_probs)

    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Victim predictions
    colors_v = ['#27ae60' if i == victim_pred else '#95a5a6' for i in range(10)]
    axes[0].barh(range(10), victim_probs, color=colors_v)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(CLASS_NAMES)
    axes[0].set_xlim(0, 1)
    axes[0].set_title(f'Victim Model\nPrediction: {CLASS_NAMES[victim_pred]}', fontweight='bold')
    axes[0].invert_yaxis()

    # Stolen predictions
    colors_s = ['#c0392b' if i == stolen_pred else '#95a5a6' for i in range(10)]
    axes[1].barh(range(10), stolen_probs, color=colors_s)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f'Stolen Model\nPrediction: {CLASS_NAMES[stolen_pred]}', fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()

    # Convert to image using BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_image = np.array(Image.open(buf))
    buf.close()
    plt.close()

    # Agreement status
    agreement = "MATCH" if victim_pred == stolen_pred else "MISMATCH"

    victim_result = f"{CLASS_NAMES[victim_pred]} ({victim_probs[victim_pred]*100:.1f}%)"
    stolen_result = f"{CLASS_NAMES[stolen_pred]} ({stolen_probs[stolen_pred]*100:.1f}%)"

    return plot_image, agreement, victim_result, stolen_result


def random_sample_and_predict():
    """Get random sample and run prediction."""
    img, true_label, _ = get_random_sample()
    plot, agreement, victim_res, stolen_res = predict_and_compare(img)
    return img, plot, agreement, victim_res, stolen_res, true_label


# Build Gradio interface
load_resources()

with gr.Blocks(title="Model Extraction Attack Demo") as demo:
    gr.Markdown("""
    # Model Extraction Attack Demo

    This demo shows how an attacker can **steal** an AI model's functionality by training a substitute model
    that mimics the original. The "stolen" model was trained using only 1,000 queries to the victim model,
    without any access to the original training data.

    **Try it yourself:** Upload an image or click "Random Sample" to see how closely the stolen model
    matches the victim's predictions.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="pil", height=200)
            with gr.Row():
                random_btn = gr.Button("Random Sample", variant="primary")
                clear_btn = gr.Button("Clear")
            true_label = gr.Textbox(label="True Label (for random samples)", interactive=False)

        with gr.Column(scale=2):
            output_plot = gr.Image(label="Prediction Comparison", height=300)
            with gr.Row():
                agreement_box = gr.Textbox(label="Agreement", interactive=False)
                victim_pred = gr.Textbox(label="Victim Prediction", interactive=False)
                stolen_pred = gr.Textbox(label="Stolen Prediction", interactive=False)

    gr.Markdown("""
    ---
    ### How the attack works:
    1. **Query Phase**: Attacker sends images to the victim model's API and records predictions
    2. **Training Phase**: Attacker trains their own model to match the victim's outputs
    3. **Result**: A "stolen" model that behaves like the original, built at a fraction of the cost

    *The stolen model in this demo achieves ~90% agreement with the victim model using only 1,000 queries.*
    """)

    # Event handlers
    input_image.change(
        fn=predict_and_compare,
        inputs=[input_image],
        outputs=[output_plot, agreement_box, victim_pred, stolen_pred]
    )

    random_btn.click(
        fn=random_sample_and_predict,
        outputs=[input_image, output_plot, agreement_box, victim_pred, stolen_pred, true_label]
    )

    clear_btn.click(
        fn=lambda: (None, None, "", "", "", ""),
        outputs=[input_image, output_plot, agreement_box, victim_pred, stolen_pred, true_label]
    )


if __name__ == "__main__":
    demo.launch()
