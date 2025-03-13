# import streamlit as st
# import pickle
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import csr_matrix

# # Load Data and Models
# df = pd.read_csv('df.csv', encoding='latin1')
# books_df = pd.read_csv('Books.csv', encoding='latin1')
# df = pd.merge(df, books_df[['ISBN', 'Image-URL-M']], on='ISBN', how='left')

# with open('popularity_model.pkl', 'rb') as file:
#     popular_books = pickle.load(file)
# with open('knn_model.pkl', 'rb') as file:
#     knn = pickle.load(file)
# with open('content_based_model.pkl', 'rb') as file:
#     content_based_data = pickle.load(file)
#     tfidf = content_based_data['tfidf_model']
#     book_similarity = content_based_data['book_similarity_matrix']
#     book_index = content_based_data['book_index']
# # with open('original_user_item_matrix.pkl', 'rb') as f:
# #     original_user_item_matrix = pickle.load(f)
# with open('user_item_matrix.pkl', 'rb') as f:
#     user_item_matrix = pickle.load(f)

# # Recommendation Functions
# def get_popular_recommendations(top_n=10):
#     return popular_books.head(top_n)

# def get_knn_recommendations_for_user(user_id, n=10):
#     if user_id not in user_item_matrix.index:
#         return "User not found in the matrix!"
#     user_idx = user_item_matrix.index.get_loc(user_id)
#     distances, indices = knn.kneighbors(csr_matrix(user_item_matrix.iloc[user_idx]), n_neighbors=n+1)
#     similar_users_isbns = user_item_matrix.iloc[indices.flatten()[1:]].columns.tolist()
#     recommended_books = df[df["ISBN"].isin(similar_users_isbns)][
#         ["Book-Title", "Book-Author", "Book-Rating", "Image-URL-M"]
#     ].drop_duplicates()
#     top_10_books = recommended_books.sort_values(by=["Book-Rating"], ascending=False).head(n)
#     return top_10_books[["Book-Title", "Book-Author", "Image-URL-M"]]

# def get_content_based_recommendations(book_title, n=15):
#     if book_title not in book_index:
#         return "Book not found!"
#     idx = book_index[book_title]
#     if idx >= len(book_similarity):
#         return "Book index out of range!"
#     similar_books = book_similarity[idx].argsort()[-n-1:-1][::-1]
#     return df.iloc[similar_books][["Book-Title", "Book-Author", "Image-URL-M"]].drop_duplicates()

# # Streamlit App
# st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

# # Custom CSS for enhanced styling
# st.markdown(
#     """
#     <style>
#     .big-font { font-size:36px !important; font-weight: bold; color: #3366cc; text-align: center;}
#     .medium-font { font-size: 24px !important; color: #555; text-align: center;}
#     .recommendation-card { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center; background-color: #f9f9f9; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
#     .recommendation-card h4 { color: #333; margin-bottom: 5px; }
#     .recommendation-card p { color: #666; margin-bottom: 10px; }
#     .stRadio > label { margin-right: 20px; }
#     .stRadio { display: flex; justify-content: center; }
#     .st-eb {background-color: #f0f8ff;}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown('<p class="big-font">📚 Book Recommendation System 📚</p>', unsafe_allow_html=True)

# # Model Selection
# st.markdown('<p class="medium-font">Select Recommendation Model</p>', unsafe_allow_html=True)
# model_selection = st.radio(
#     "",
#     ["Popularity-Based", "Collaborative Filtering (KNN)", "Content-Based Filtering"],
#     horizontal=True,
#     label_visibility="collapsed"
# )

# # Recommendation Display
# def display_recommendations_grid(recommendations):
#     if isinstance(recommendations, str):
#         st.error(recommendations)
#     else:
#         cols = st.columns(5)
#         for index, row in recommendations.reset_index(drop=True).iterrows():
#             col_index = index % 5
#             with cols[col_index]:
#                 st.markdown(f"""
#                     <div class="recommendation-card">
#                         <h4>{row['Book-Title']}</h4>
#                         <p>Author: {row['Book-Author']}</p>
#                 """, unsafe_allow_html=True)
#                 if 'Image-URL-M' in row:
#                     st.markdown(f"""
#                         <div style="display: flex; justify-content: center;">
#                             <img src="{row['Image-URL-M']}" width="120">
#                         </div>
#                     """, unsafe_allow_html=True)
#                 st.markdown("</div>", unsafe_allow_html=True)

# # Main Content Area
# st.markdown("---")

# if model_selection == "Popularity-Based":
#     st.markdown('<p class="medium-font">Popularity-Based Recommendations</p>', unsafe_allow_html=True)
#     recommendations = get_popular_recommendations()
#     st.table(recommendations)

# elif model_selection == "Collaborative Filtering (KNN)":
#     st.markdown('<p class="medium-font">Collaborative Filtering (KNN) Recommendations</p>', unsafe_allow_html=True)
#     user_id = st.number_input("Enter User ID:", min_value=1, step=1, value=278221)
#     if user_id:
#         recommendations = get_knn_recommendations_for_user(user_id)
#         display_recommendations_grid(recommendations)

# elif model_selection == "Content-Based Filtering":
#     st.markdown('<p class="medium-font">Content-Based Filtering Recommendations</p>', unsafe_allow_html=True)
#     book_title = st.text_input("Enter a book title:", value="The Da Vinci Code")
#     if book_title:
#         recommendations = get_content_based_recommendations(book_title)
#         display_recommendations_grid(recommendations)



import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load Data
df = pd.read_csv('E:\\excelR\\project3\\df.csv', encoding='latin1')
books_df = pd.read_csv('E:\\excelR\\project3\\Books.csv', encoding='latin1')

# Merge with book images
df = pd.merge(df, books_df[['ISBN', 'Image-URL-M']], on='ISBN', how='left')

# Load Pickle Models
with open('E:\\excelR\\project3\\popularity_model.pkl', 'rb') as file:
    popular_books = pickle.load(file)
with open('E:\\excelR\\project3\\knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)
with open('E:\\excelR\\project3\\content_based_model.pkl', 'rb') as file:
    content_based_data = pickle.load(file)
    tfidf = content_based_data['tfidf_model']
    book_similarity = content_based_data['book_similarity_matrix']
    book_index = content_based_data['book_index']
with open('E:\\excelR\\project3\\user_item_matrix.pkl', 'rb') as f:
    user_item_matrix = pickle.load(f)

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
