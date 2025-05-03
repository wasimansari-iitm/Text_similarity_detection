# 🤖 Matchy: Text Similarity Detector

Welcome to **Matchy** — an AI-powered application for detecting similarity between pairs of text using fuzzy logic and semantic analysis using word2vec! Built with ❤️ for NLP enthusiasts, this app is optimized to work with real-world text data like Quora question pairs.

[![Docker](https://img.shields.io/badge/Built%20With-Docker-blue?logo=docker)](https://www.docker.com/)
[![GitHub Repo](https://img.shields.io/badge/Source-GitHub-black?logo=github)](https://github.com/wasimansari-iitm/Text_similarity_detection)

---

## 🔍 What is Matchy?

**Matchy** is a lightweight yet powerful service that identifies whether two given pieces of text are duplicates or not. It uses:

- ✨ **Preprocessing pipelines** for cleaning and standardizing input
- 🧠 **Word embeddings + Fuzzy logic** for smart feature extraction
- 📈 **Gradient boosting (XGBoost)** for final prediction
- 🔁 **Dockerized deployment** for portability and ease-of-use

---

## ✨ Features

- 📝 **Compare Two Questions**  
  Input two questions and find out if they are semantically the same.

- 📈 **Probability Score**  
  Along with "Duplicate" or "Not Duplicate", Matchy returns a confidence probability.

- 📜 **History Log**  
  Keeps track of all questions asked in the current session.

- 🧠 **Smart ML Pipeline**  
  Combines:
  - Preprocessing
  - 21 custom-engineered features
  - Word2Vec embeddings
  - XGBoost classifier

- 🌙 **Modern Dark UI**  
  Intuitive, responsive frontend with live feedback and elegant styling.

- 🐳 **Dockerized for Easy Deployment**  
  Run the app with just a few Docker commands (see below).

---

## 📦 Installation (via Docker)

To get started, make sure you have Docker installed. Then, run the following:

```bash
# Clone the repo
git clone https://github.com/wasimansari-iitm/Text_similarity_detection.git
cd Text_similarity_detection

# Build the Docker image
docker build -t matchy .

# Run the container
docker run -p 7860:7860 matchy
````

The API will be available at:
👉 **[http://localhost:7860/predictor](http://localhost:7860/predictor)**

---

## 🧪 API Usage

### POST `/predictor`

**Request Body:**

```json
{
  "q1": "How do I learn Python?",
  "q2": "What is the best way to study Python?"
}
```

**Response:**

```json
{
  "prediction": "Duplicate",
  "Probability": 0.66
}
```

---

## 🛠 Project Structure

```bash
├── app.py              # Flask backend
├── helper.py           # Feature engineering logic
├── model               # Pre-trained XGBoost model
├── Dockerfile          # For containerization
└── README.md           # You're here!
```

---

## ✍️ Author

Made with ❤️ by **Wasim Ansari**
📍 Data Science | IIT Madras
🌐 [GitHub](https://github.com/wasimansari-iitm)

---

🤝 Acknowledgements

    ✨ Built using GitHub Copilot + ChatGPT + your perseverance.

    💙 Inspired by the Quora Duplicate Question Detection problem.

    🌟 Designed with simplicity and elegance in mind.

---

## 📃 License

This project is open-source under the [MIT License](LICENSE).

---

## ⭐ Show Your Support

If you find this project useful, please give it a ⭐ on GitHub! It helps more people discover **Matchy**.