# Number Identification – Handwritten Digit Recognition

Welcome to Number Identification, a fun and interactive project where you can draw a number on the screen and watch a neural network guess what it is. This project combines the power of deep learning with a simple user interface to recognize digits written by hand — just like the MNIST dataset it's trained on.

Built using Python, TensorFlow, and a lightweight Tkinter GUI, this app is a practical demonstration of computer vision and neural networks applied in a real-time environment.

---

# What It Does

- Opens a window where users can draw a digit (0–9) using the mouse.
- Preprocesses the drawing (resizing, inversion, normalization).
- Feeds it into a trained neural network based on the MNIST dataset.
- Outputs the predicted digit with high accuracy.

---

# Tech Stack

- Python 3
- TensorFlow / Keras – for model building and training
- Tkinter** – to create the GUI window
- NumPy & Pillow (PIL) – for image manipulation

---

### 1. Clone the repository

```bash
git clone https://github.com/KathanParikh/Number_Identification.git
cd Number_Identification
