import streamlit as st
import pickle
import pandas as pd
import requests
from io import BytesIO
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url, encoding='latin1')

def load_pickle_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

# Load Data from Google Drive
df = load_csv_from_drive("1ACsDx6V8k19J-S63LSseWYRp_rsM-CG3")
books_df = load_csv_from_drive("1miWbT_4ltE9uuPPSS2nBRMPNiExyT0Bj")
ratings = load_csv_from_drive("1dwwXfcI9NzQVliyG-hGueqoTKCDe6G1j")
users = load_csv_from_drive("1hqaiWx-2Ht_9kWrwGcWGcSu4Y1RO-e95")

# Merge with book images
df = pd.merge(df, books_df[['ISBN', 'Image-URL-M']], on='ISBN', how='left')

# Load Pickle Models from Google Drive
popular_books = load_pickle_from_drive("1f2BNeiE3_8Y9akIvB96QgqgG9JdtiDdQ")
knn = load_pickle_from_drive("1m-45jM6Q4c32DeyiSEhfAPuUIhohlKSX")
content_based_data = load_pickle_from_drive("1nbDPLTxckiJnJp8FwFHP1fXyUfyNBsVl")
user_item_matrix = load_pickle_from_drive("1enD9Rjtbu_zXJFDNtPYxsK4l6Q4ZNaFC")

# Extract Content-Based Filtering Components
tfidf = content_based_data['tfidf_model']
book_similarity = content_based_data['book_similarity_matrix']
book_index = content_based_data['book_index']

def get_popular_recommendations(top_n=10):
    return popular_books.head(top_n)

def get_knn_recommendations_for_user(user_id, n=10):
    if user_id not in user_item_matrix.index:
        return "User not found in the matrix!"
    user_idx = user_item_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors(csr_matrix(user_item_matrix.iloc[user_idx]), n_neighbors=n+1)
    similar_users_isbns = user_item_matrix.iloc[indices.flatten()[1:]].columns.tolist()
    recommended_books = df[df["ISBN"].isin(similar_users_isbns)][["Book-Title", "Book-Author", "Image-URL-M"]].drop_duplicates()
    return recommended_books.head(n)

def get_content_based_recommendations(book_title, n=15):
    if book_title not in book_index:
        return "Book not found!"
    idx = book_index[book_title]
    similar_books = book_similarity[idx].argsort()[-n-1:-1][::-1]
    return df.iloc[similar_books][["Book-Title", "Book-Author", "Image-URL-M"]].drop_duplicates()

# Streamlit UI
st.title("📚 Book Recommendation System")
st.sidebar.header("Select a Recommendation Type")

option = st.sidebar.selectbox(
    "Choose a recommendation method:",
    ("Popularity-Based", "User-Based Collaborative Filtering", "Content-Based")
)

if option == "Popularity-Based":
    st.subheader("🔥 Popular Books")
    num_books = st.slider("Number of books to show", 5, 20, 10)
    recommendations = get_popular_recommendations(num_books)
    
    for _, row in recommendations.iterrows():
        st.write(f"📖 **{row['Book-Title']}** by {row['Book-Author']}")

elif option == "User-Based Collaborative Filtering":
    st.subheader("👤 Personalized Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    if st.button("Get Recommendations"):
        recommendations = get_knn_recommendations_for_user(user_id)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for _, row in recommendations.iterrows():
                st.image(row["Image-URL-M"], width=100)
                st.write(f"📖 **{row['Book-Title']}** by {row['Book-Author']}")

elif option == "Content-Based":
    st.subheader("📖 Content-Based Recommendations")
    book_title = st.text_input("Enter a book title")
    if st.button("Find Similar Books"):
        recommendations = get_content_based_recommendations(book_title)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for _, row in recommendations.iterrows():
                st.image(row["Image-URL-M"], width=100)
                st.write(f"📖 **{row['Book-Title']}** by {row['Book-Author']}")
