# 🛍️ E-Commerce Product Recommendation System

A machine learning-powered recommendation engine that suggests products to users based on their historical behavior. This project uses Collaborative Filtering via Matrix Factorization (Singular Value Decomposition) to predict user-item interactions.

## 🚀 Features
* **Matrix Factorization (SVD):** Highly optimized collaborative filtering model.
* **Offline Evaluation:** Calculates Precision@K, Recall@K, and NDCG@K to ensure ranking quality.
* **Streamlit UI:** A clean, interactive web interface to simulate real-time user recommendations.
* **Cold Start Handling:** Gracefully handles brand-new users by defaulting to popular items.

### 📂 Directory Structure
Create a folder named ```ecommerce-recommender``` and arrange your files like this:
```
ecommerce-recommender/
│
├── app.py                  # The Streamlit web app
├── train_and_evaluate.py   # The data generation and training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
## 💻 Usage

**Step 1: Train the model and generate data**
Run the training script to generate the synthetic e-commerce dataset, train the SVD model, evaluate metrics, and save the artifacts (`recommender_model.pkl` and `products.csv`).

```bash
python train_and_evaluate.py

```

**Step 2: Launch the web app**
Start the Streamlit dashboard to interact with the model.

```bash
streamlit run app.py

```
