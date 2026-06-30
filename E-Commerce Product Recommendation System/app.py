import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 1. LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_artifacts():
    with open('recommender_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['trainset']

@st.cache_data
def load_products():
    return pd.read_csv('products.csv')

model, trainset = load_artifacts()
products_df = load_products()

# ==========================================
# 2. RECOMMENDATION LOGIC
# ==========================================
def get_top_n_recommendations(user_id, n=5):
    # Get a list of all item IDs
    all_item_ids = products_df['item_id'].unique()
    
    # Check if user exists in the training set
    try:
        inner_user_id = trainset.to_inner_uid(user_id)
        # Get items the user has already interacted with
        user_interacted_items = set([j for (j, _) in trainset.ur[inner_user_id]])
        user_interacted_raw_ids = [trainset.to_raw_iid(inner_id) for inner_id in user_interacted_items]
    except ValueError:
        # User is brand new (Cold Start) - we'll just recommend items anyway
        user_interacted_raw_ids = []

    # Predict ratings for all items the user HAS NOT interacted with
    predictions = []
    for item_id in all_item_ids:
        if item_id not in user_interacted_raw_ids:
            est_rating = model.predict(uid=user_id, iid=item_id).est
            predictions.append((item_id, est_rating))
            
    # Sort by highest predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N item IDs
    top_n_item_ids = [item_id for item_id, rating in predictions[:n]]
    return top_n_item_ids

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Store Recommender", layout="centered")

st.title("🛍️ E-Commerce Recommender System")
st.write("Using Collaborative Filtering & Matrix Factorization")
st.markdown("---")

# User Input
user_id = st.number_input("Enter your User ID (1 - 100):", min_value=1, max_value=100, step=1, value=15)

if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Crunching the numbers..."):
        top_items = get_top_n_recommendations(user_id=user_id, n=5)
        
        st.subheader(f"Top 5 Product Picks for User {user_id}")
        
        # Display as cards
        cols = st.columns(5)
        for idx, item_id in enumerate(top_items):
            # Fetch product details
            product_info = products_df[products_df['item_id'] == item_id].iloc[0]
            
            with cols[idx]:
                # Placeholder for an actual product image
                st.image("https://placehold.co/150x150/e2e8f0/1e293b?text=Item+" + str(item_id), use_container_width=True)
                st.markdown(f"**{product_info['product_name']}**")
                st.caption(f"{product_info['category']}")
                st.write(f"**${product_info['price']}**")