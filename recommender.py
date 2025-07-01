import faiss, numpy as np, torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def _load_cnn():
    mdl = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    mdl.fc = torch.nn.Identity()
    return mdl.to(DEVICE).eval()

@torch.inference_mode()
def _preprocess(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    return tfm(img).unsqueeze(0).to(DEVICE)

class FashionRecommender:
    def __init__(self, model_dir="models", k=6):
        self.k = k
        self.paths = np.load(Path(model_dir)/"image_paths.npy")
        self.index = faiss.read_index(str(Path(model_dir)/"faiss_index.bin"))
        self.model = _load_cnn()

    def extract(self, img: Image.Image) -> np.ndarray:
        feat = self.model(_preprocess(img)).cpu().numpy().flatten()
        return (feat / np.linalg.norm(feat)).astype("float32")

    def search(self, img: Image.Image):
        q = self.extract(img)[None, :]
        _, I = self.index.search(q, self.k + 1)
        return [self.paths[i] for i in I[0][1:]]
