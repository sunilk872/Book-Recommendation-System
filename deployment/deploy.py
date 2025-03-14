import streamlit as st
import pickle
import pandas as pd
import requests
from io import BytesIO
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Set Streamlit page configuration
st.set_page_config(page_title="Book Recommendation System 📖", layout="wide")


# Function to Load CSV from Google Drive
def load_csv_from_drive(url):
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    return pd.read_csv(BytesIO(response.content), encoding="latin1")

# Function to Load Pickle Files from Google Drive
def load_pickle_from_drive(url):
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    return pickle.loads(response.content)

# Load Data
df = load_csv_from_drive("https://drive.google.com/file/d/1ACsDx6V8k19J-S63LSseWYRp_rsM-CG3/view?usp=sharing")
books_df = load_csv_from_drive("https://drive.google.com/file/d/1miWbT_4ltE9uuPPSS2nBRMPNiExyT0Bj/view?usp=sharing")

# Merge with book images
df = pd.merge(df, books_df[['ISBN', 'Image-URL-M']], on='ISBN', how='left')

# Load Pickle Models
popular_books = load_pickle_from_drive("https://drive.google.com/file/d/1f2BNeiE3_8Y9akIvB96QgqgG9JdtiDdQ/view?usp=sharing")
knn = load_pickle_from_drive("https://drive.google.com/file/d/1m-45jM6Q4c32DeyiSEhfAPuUIhohlKSX/view?usp=sharing")
content_based_data = load_pickle_from_drive("https://drive.google.com/file/d/1nbDPLTxckiJnJp8FwFHP1fXyUfyNBsVl/view?usp=sharing")
user_item_matrix = load_pickle_from_drive("https://drive.google.com/file/d/1enD9Rjtbu_zXJFDNtPYxsK4l6Q4ZNaFC/view?usp=drive_link")

# Extract Content-Based Data
tfidf = content_based_data['tfidf_model']
book_similarity = content_based_data['book_similarity_matrix']
book_index = {k.lower(): v for k, v in content_based_data['book_index'].items()}  # Lowercase for case-insensitivity

# Recommendation Functions
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
    book_title_lower = book_title.lower()
    if book_title_lower not in book_index:
        return "Book not found!"
    idx = book_index[book_title_lower]
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
        st.write(f"📖 *{row['Book-Title']}* by {row['Book-Author']}")

elif option == "User-Based Collaborative Filtering":
    st.subheader("👤 Personalized Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, step=1, value=388)  # Default user_id set to 388
    if st.button("Get Recommendations"):
        recommendations = get_knn_recommendations_for_user(user_id)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for _, row in recommendations.iterrows():
                st.image(row["Image-URL-M"], width=100)
                st.write(f"📖 *{row['Book-Title']}* by {row['Book-Author']}")

elif option == "Content-Based":
    st.subheader("📖 Content-Based Recommendations")
    book_title = st.text_input("Enter a book title", value="The Da Vinci Code")  # Default book title set
    if st.button("Find Similar Books"):
        recommendations = get_content_based_recommendations(book_title)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for _, row in recommendations.iterrows():
                st.image(row["Image-URL-M"], width=100)
                st.write(f"📖 *{row['Book-Title']}* by {row['Book-Author']}")
