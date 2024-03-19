## GADoT DDoS Attack Detection on LUCID
This project analyzes the effectiveness of the GADoT (GAN-based Adversarial Training for robust DDoS attack detection) approach in improving the robustness of the LUCID (Lightweight, Usable CNN in DDoS Detection) model against adversarial attacks. The study focuses on perturbing individual features in the test samples and evaluating the model's false negative rate (FNR) and F1 score before and after GADoT adversarial training.

### Methodology
Two datasets were used for training and evaluation:
CSE-CIC-IDS2018 dataset for training
CIC-IDS2017 dataset for evaluation
20 significant features were extracted using a custom feature extractor
The LUCID model was trained using the grid search technique with Adam optimizer for 50 epochs
WGAN was used to generate fake benign samples for perturbing DDoS samples
The adversarially trained LUCID model was evaluated by perturbing the following features in the test dataset:
Flow Duration
Packet Length Mean
Flow Packets per second
Flow bytes per second
