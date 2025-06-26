import tensorflow as tf
from tensorflow import keras
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load MNIST dataset
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape for CNN: (samples, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build a CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save('mnist_model.h5')
    print('CNN model trained and saved as mnist_model.h5')

class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title('Number Recognition')
        self.resizable(False, False)
        self.canvas_width = 280
        self.canvas_height = 280
        self.model = model
        self.result_var = tk.StringVar()
        self._init_widgets()
        self._init_draw()

    def _init_widgets(self):
        # Create main frame
        main_frame = tk.Frame(self)
        main_frame.pack(pady=10, padx=10)
        
        # Left side - drawing canvas
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.set_last_xy)

        predict_btn = tk.Button(left_frame, text='Predict', command=self.predict)
        predict_btn.pack(pady=5)
        clear_btn = tk.Button(left_frame, text='Clear', command=self.clear)
        clear_btn.pack(pady=5)

        result_label = tk.Label(left_frame, textvariable=self.result_var, font=('Arial', 18))
        result_label.pack(pady=5)
        
        # Right side - matplotlib plot
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas_widget.get_tk_widget().pack()
        
        # Initialize empty plot
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1)
        self.canvas_widget.draw()

    def _init_draw(self):
        self.image1 = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw1 = ImageDraw.Draw(self.image1)
        self.last_x, self.last_y = None, None

    def set_last_xy(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw1.line([self.last_x, self.last_y, event.x, event.y], fill='black', width=15)
        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        self.canvas.delete('all')
        self._init_draw()
        self.result_var.set('')
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1)
        self.canvas_widget.draw()

    def preprocess(self):
        # Convert to numpy array
        img = np.array(self.image1)
        img = 255 - img
        coords = np.column_stack(np.where(img > 20))
        if coords.size:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img[y0:y1+1, x0:x1+1]
        else:
            img = img[0:1, 0:1]
        img = Image.fromarray(img).resize((20, 20), Image.LANCZOS)
        new_img = Image.new('L', (28, 28), 0)
        upper_left = ((28 - 20) // 2, (28 - 20) // 2)
        new_img.paste(img, upper_left)
        img = np.array(new_img)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)  # For CNN
        return img

    def show_debug_image(self, img):
        pass  # No debug image

    def predict(self):
        img = self.preprocess()
        pred = self.model.predict(img)[0]
        digit = np.argmax(pred)
        confidence = pred[digit] * 100
        
        # Clear and update the embedded plot
        self.ax.clear()
        bars = self.ax.bar(range(10), pred, color=['#4caf50' if i == digit else '#2196f3' for i in range(10)])
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title(f'Prediction Probabilities (Predicted: {digit})')
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1)
        
        # Annotate bars with probability values
        for i, bar in enumerate(bars):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pred[i]:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        self.fig.tight_layout()
        self.canvas_widget.draw()
        self.result_var.set(f'Prediction: {digit} ({confidence:.2f}%)')


def run_gui():
    model = keras.models.load_model('mnist_model.h5')
    app = App(model)
    app.mainloop()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        run_gui()
    else:
        train_and_save_model()
