# 🧠🤟 AI-Powered ASL Translator

**Translate American Sign Language into text using AI — real-time via webcam or through uploaded images. Train and compare ML models easily with a beautiful web interface.**

> Bridging the gap between silence and speech using Computer Vision and Machine Learning.

**🔗 Live Demo:** [asl-translator-x4wf.onrender.com](https://asl-translator-x4wf.onrender.com)  
**📁 GitHub Repo:** [github.com/miyasajid19/ASL-Translator](https://github.com/miyasajid19/ASL-Translator.git)

![image](https://github.com/user-attachments/assets/2b68dbc6-43f2-4baf-a7b2-9ad75ab7c6cc)
  

---

## ✨ Features

- **📹 Real-Time Translation:** Translate ASL signs via webcam using MediaPipe + Scikit-learn.
- **🖼️ Image-Based Detection:** Upload multiple images and recognize ASL signs with your trained model.
- **🧠 Train Custom Models:** Choose from ML classifiers like Random Forest, SVM, Logistic Regression, etc.
- **📊 Compare Algorithms:** Evaluate multiple classifiers on the same dataset with interactive visualizations.
- **📈 Insightful Charts:** Built-in Plotly and Chart.js integration for data insights.
- **🧩 Modular Design:** Easy to extend or modify any component — backend, models, or UI.

---

## 🧰 Tech Stack

**Backend**
- Python 3.8+
- Flask + Flask-SocketIO
- OpenCV
- MediaPipe (Google)
- Scikit-learn
- Pandas, NumPy

**Frontend**
- HTML, CSS (Bootstrap 5)
- JavaScript (jQuery, vanilla)
- Plotly.js & Chart.js
- Socket.IO Client

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip`
- Git
- Webcam (for RTP)

### Installation

```bash
git clone https://github.com/miyasajid19/ASL-Translator.git
cd ASL-Translator

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Secret Key Setup

```bash
# macOS/Linux
export SECRET_KEY="your_random_secure_key"
# Windows CMD
set SECRET_KEY=your_random_secure_key
```

### Run the App

```bash
python main.py
# Visit http://localhost:5000
```

---

## 🗂️ Folder Structure

```
ASL-Translator/
├── main.py
├── requirements.txt
├── static/
│   └── files/
│       ├── datasets/    # Input datasets (.pkl)
│       └── models/      # Trained models (.pkl)
├── templates/
│   ├── index.html
│   ├── real_time_processing.html
│   ├── image_processing.html
│   ├── train.html
│   └── compare_models.html
└── README.md
```

---

## 🧪 How to Use

### 🔴 Real-Time ASL (`/RTP`)
- Choose a trained model.
- Start webcam.
- Perform ASL signs — see predictions live and track recognized sequence.

### 🖼️ Image Upload (`/ImageProcessing`)
- Choose a model.
- Upload multiple images.
- See individual predictions + combined sequence output.

### 📚 Train Model (`/train`)
- Select dataset + algorithm.
- Enter a unique model name.
- View accuracy, training time, plots — model is saved.

### 📊 Compare Models (`/compare`)
- Choose a dataset + multiple algorithms.
- Get detailed metrics (accuracy, precision, F1-score).
- Compare results in charts & tables.

---

## 📦 Data Format

### Datasets (`.pkl`)
```python
{
  'data': [list of 42 floats],  # Flattened (x,y) hand landmarks
  'label': ['A', 'B', 'Hello', ...]
}
```

### Models (`.pkl`)
```python
{
  'model': sklearn classifier,
  'accuracy': float,
  'algorithm_key': str,
  ...
}
```

---

## ⚙️ Configuration Highlights

In `main.py`:

```python
SECRET_KEY = os.environ.get("SECRET_KEY")
MODEL_DIR = './static/files/models/'
DATASET_DIR = './static/files/datasets/'
EXPECTED_LANDMARK_FEATURES = 42
NO_HAND_TIMEOUT = 2.0
```

---

## 🛣 Roadmap

- [x] Webcam-based static ASL translation
- [x] Image batch upload
- [x] Model training & saving
- [x] Comparison of classifiers
- [ ] Dynamic sign recognition (word-level or sentence-level)
- [ ] Dataset creation module
- [ ] Docker deployment support
- [ ] Speech output integration (TTS)

---

## 🤝 Contributing

```bash
# Fork -> Clone -> Create Branch -> Commit -> PR
git checkout -b feature/your-feature-name
```

Please follow PEP8 and document major changes. Open issues for discussions.

---

## ⚖️ License

Licensed under the **MIT License**.  
See [`LICENSE`](./LICENSE) for more details.

---

## 🙏 Credits

- [MediaPipe](https://mediapipe.dev/)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [Bootstrap](https://getbootstrap.com/)
- [Plotly](https://plotly.com/)
- [Chart.js](https://www.chartjs.org/)

---

Made with ❤️ by **Sajid Miya**

[🔗 Live App](https://asl-translator-x4wf.onrender.com) • [📁 GitHub](https://github.com/miyasajid19/ASL-Translator.git)
