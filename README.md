# Hand2Sign 🤟

**Hand2Sign** is a deep learning-based application that translates hand gestures into sign language, aiming to bridge the communication gap between the hearing and speech impaired and the general population. It leverages advanced computer vision techniques and neural networks to recognize real-time hand gestures and convert them into text or speech.

## 🚀 Features

- 🖐️ Real-time hand gesture recognition using webcam
- 🤖 Deep learning model trained on sign language datasets
- 🔤 Translation of gestures into English text
- 🎤 Optionally converts recognized text into speech (TTS)
- 📊 User-friendly interface for live prediction
- 🧠 Model trained using CNN + Bi-LSTM for spatial-temporal analysis

## 📁 Project Structure

```
hand2sign/
├── MP_Data/     # Preprocessed dataset for trainin
├── static/      # css files
├── templates/   # html files
├── model/       # Saved model weights and architecture
├── sign.py/     # python code for training and experiments
├── app/         # Flask app for UI
├── utils/       # Helper scripts and preprocessing tools
├── requirements.txt # Python dependencies
└── README.md    # Project documentation
```

## 🔧 Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/sah1l-17/hand2sign.git
   cd hand2sign
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   python app.py
   ```

## 🧠 Model Overview

The model is built using:

- Convolutional Neural Networks (CNNs) for spatial feature extraction
- Bi-directional LSTM (Bi-LSTM) for temporal sequence learning
- Trained on the Custom dataset

## 📹 Demo

Coming soon: video demo of real-time prediction

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- PyTorch / TensorFlow
- Flask
- NumPy, Pandas, Matplotlib


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Sahil Ansari – [@sah1l-17](https://github.com/sah1l-17)
