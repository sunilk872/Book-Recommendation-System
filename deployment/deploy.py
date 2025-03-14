import streamlit as st
import pickle
import pandas as pd
import requests
from io import BytesIO
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Set Streamlit page configuration
st.set_page_config(
    page_title="Book Recommendation System 📖",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Clean professional styling */
    .main {background-color: #f8f9fa}
    .header {color: #2c3e50; padding: 2rem; background: white; border-radius: 15px; margin-bottom: 2rem}
    .book-card {background: white; border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1)}
    .sidebar {background: white !important; border-radius: 15px; padding: 1.5rem !important}
    .stButton>button {background-color: #4a90e2 !important; color: white !important}
</style>
""", unsafe_allow_html=True)

# Data loading functions
def load_csv_from_drive(url):
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(BytesIO(requests.get(download_url).content), encoding="latin1")

def load_pickle_from_drive(url):
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    return pickle.loads(requests.get(download_url).content)

# Load datasets first
@st.cache_resource
def load_data():
    df = load_csv_from_drive("https://drive.google.com/file/d/1ACsDx6V8k19J-S63LSseWYRp_rsM-CG3/view")
    books_df = load_csv_from_drive("https://drive.google.com/file/d/1miWbT_4ltE9uuPPSS2nBRMPNiExyT0Bj/view")
    merged_df = pd.merge(df, books_df[['ISBN', 'Image-URL-M']], on='ISBN', how='left')
    return merged_df

@st.cache_resource
def load_models():
    return {
        'popular_books': load_pickle_from_drive("https://drive.google.com/file/d/1f2BNeiE3_8Y9akIvB96QgqgG9JdtiDdQ/view"),
        'knn': load_pickle_from_drive("https://drive.google.com/file/d/1m-45jM6Q4c32DeyiSEhfAPuUIhohlKSX/view"),
        'content_based': load_pickle_from_drive("https://drive.google.com/file/d/1nbDPLTxckiJnJp8FwFHP1fXyUfyNBsVl/view"),
        'user_matrix': load_pickle_from_drive("https://drive.google.com/file/d/1enD9Rjtbu_zXJFDNtPYxsK4l6Q4ZNaFC/view")
    }

# Function definitions after loading data
def get_popular_recommendations(top_n=10):
    return models['popular_books'].head(top_n)

def get_knn_recommendations(user_id, n=10):
    if user_id not in models['user_matrix'].index:
        return "User not found!"
    user_idx = models['user_matrix'].index.get_loc(user_id)
    distances, indices = models['knn'].kneighbors(csr_matrix(models['user_matrix'].iloc[user_idx]), n_neighbors=n+1)
    return models['user_matrix'].iloc[indices.flatten()[1:]].columns.tolist()

# Load data and models
df = load_data()
models = load_models()

# Streamlit UI
st.markdown("<div class='header'><h1>Book Recommendation System</h1></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📚 Recommendation Type")
    option = st.selectbox(
        "Choose method:",
        ("Popularity-Based", "User-Based", "Content-Based"),
        label_visibility="collapsed"
    )

# Recommendation display functions
def display_books(books_df, cols=5):
    columns = st.columns(cols)
    for idx, (_, row) in enumerate(books_df.iterrows()):
        with columns[idx % cols]:
            with st.container():
                st.markdown("<div class='book-card'>", unsafe_allow_html=True)
                st.image(row["Image-URL-M"], use_column_width=True)
                st.markdown(f"**{row['Book-Title']}**")
                st.caption(f"by {row['Book-Author']}")
                st.markdown("</div>", unsafe_allow_html=True)

# Handle recommendations
if option == "Popularity-Based":
    st.subheader("📈 Trending Books")
    num_books = st.slider("Number of books", 5, 25, 12)
    display_books(get_popular_recommendations(num_books))

elif option == "User-Based":
    st.subheader("👤 Personalized Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, value=388)
    if st.button("Get Recommendations"):
        recommendations = get_knn_recommendations(user_id)
        display_books(df[df["ISBN"].isin(recommendations)])

elif option == "Content-Based":
    st.subheader("🔍 Similar Books")
    book_title = st.text_input("Enter book title", "The Da Vinci Code")
    if st.button("Find Similar"):
        # Add content-based recommendation logic
        pass

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; margin-top: 2rem'>Book Recommendation System v2.0</div>", unsafe_allow_html=True)
