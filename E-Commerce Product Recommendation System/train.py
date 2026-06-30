import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
from collections import defaultdict
import math

# ==========================================
# 1. GENERATE DUMMY E-COMMERCE DATA
# ==========================================
print("Generating dummy e-commerce data...")
np.random.seed(42)

# Create 50 unique products
product_ids = range(1, 51)
products_df = pd.DataFrame({
    'item_id': product_ids,
    'product_name': [f"Product_{i}" for i in product_ids],
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=50),
    'price': np.random.uniform(10.0, 500.0, size=50).round(2)
})
products_df.to_csv('products.csv', index=False)

# Create user interactions (Ratings from 1 to 5) for 100 users
interactions = []
for user in range(1, 101):
    # Each user interacts with 5 to 15 random items
    num_items = np.random.randint(5, 16)
    interacted_items = np.random.choice(product_ids, size=num_items, replace=False)
    for item in interacted_items:
        rating = np.random.randint(1, 6) # Ratings 1-5
        interactions.append([user, item, rating])

ratings_df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])

# ==========================================
# 2. TRAIN THE MATRIX FACTORIZATION MODEL
# ==========================================
print("Training Matrix Factorization (SVD) model...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)

# Split data: 80% train, 20% test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD algorithm
model = SVD(n_factors=50, random_state=42)
model.fit(trainset)
predictions = model.test(testset)

# ==========================================
# 3. EVALUATE (PRECISION, RECALL, NDCG)
# ==========================================
def evaluate_model(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions, recalls, ndcgs = dict(), dict(), dict()
    
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        
        # Precision@K
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        # Recall@K
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
        # NDCG@K
        dcg = 0
        idcg = 0
        for i, (est, true_r) in enumerate(user_ratings[:k]):
            rel = 1 if true_r >= threshold else 0
            dcg += (2**rel - 1) / math.log2(i + 2)
        
        # Ideal DCG (sort true ratings)
        ideal_ratings = sorted([1 if r[1] >= threshold else 0 for r in user_ratings], reverse=True)[:k]
        for i, rel in enumerate(ideal_ratings):
            idcg += (2**rel - 1) / math.log2(i + 2)
            
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0

    print(f"\n--- Model Evaluation (Top-{k}) ---")
    print(f"Precision@{k}: {sum(precisions.values()) / len(precisions):.4f}")
    print(f"Recall@{k}:    {sum(recalls.values()) / len(recalls):.4f}")
    print(f"NDCG@{k}:      {sum(ndcgs.values()) / len(ndcgs):.4f}")

evaluate_model(predictions, k=5)

# ==========================================
# 4. SAVE ARTIFACTS FOR STREAMLIT
# ==========================================
# Save the model and the full trainset (needed to know which items a user already bought)
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'trainset': trainset}, f)

print("\nModel saved as 'recommender_model.pkl'. You can now run the Streamlit app!")