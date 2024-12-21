import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load data
try:
    data = pd.read_csv("D://Download//R7data.csv")
except FileNotFoundError:
    st.error("Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()

# ... (Phần xử lý dữ liệu - giữ nguyên) ...

# Xử lý dữ liệu: tổng hợp các ngôn ngữ, thể loại và nền tảng
aggregated_data = (
    data.groupby(['ID_GAME', 'TEN_GAME', 'NGAY_PHAT_HANH', 'GIA', 'DANH_GIA',
                 'TY_LE_DANH_GIA_TICH_CUC', 'SO_LUONG_DANH_GIA', 'NHA_PHAT_HANH'])
    .agg({
        'TEN_THE_LOAI': lambda x: ', '.join(sorted(set(x))),
        'TEN_NGON_NGU': lambda x: ', '.join(sorted(set(x))),
        'TEN_NEN_TANG': lambda x: ', '.join(sorted(set(x)))
    })
    .reset_index()
)

# Loại bỏ các cột không cần thiết
aggregated_data = aggregated_data.drop(['ID_GAME', 'NGAY_PHAT_HANH', 'TY_LE_DANH_GIA_TICH_CUC', 'SO_LUONG_DANH_GIA'],
                                      axis=1)


# Mã hóa One-Hot Encoding
def one_hot_encode_column(df, column):
    exploded = df[column].str.split(', ').explode()
    onehot = pd.get_dummies(exploded, prefix=column)
    onehot = onehot.groupby(level=0).sum()
    return onehot


onehot_encoded_genres = one_hot_encode_column(aggregated_data, 'TEN_THE_LOAI')
onehot_encoded_languages = one_hot_encode_column(aggregated_data, 'TEN_NGON_NGU')
onehot_encoded_platforms = one_hot_encode_column(aggregated_data, 'TEN_NEN_TANG')

# Kết hợp các cột One-Hot Encoding
aggregated_data = pd.concat([
    aggregated_data.drop(['TEN_THE_LOAI', 'TEN_NGON_NGU', 'TEN_NEN_TANG'], axis=1),
    onehot_encoded_genres,
    onehot_encoded_languages,
    onehot_encoded_platforms
], axis=1)

# Chuẩn hóa cột số
scaler = StandardScaler()
numerical_features = ['GIA']
aggregated_data[numerical_features] = scaler.fit_transform(aggregated_data[numerical_features])

# Tạo ma trận đặc trưng cho mô hình KNN
features = aggregated_data.drop(['TEN_GAME', 'NHA_PHAT_HANH', 'DANH_GIA'], axis=1)

# Xây dựng mô hình KNN
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(features)


# Hàm gợi ý game
def recommend_games_knn(game_name, data, model, top_n=5):
    try:
        # Tìm chỉ số game
        idx = data[data['TEN_GAME'] == game_name].index[0]
        # Tìm các hàng xóm gần nhất
        distances, indices = model.kneighbors(features.iloc[[idx]], n_neighbors=top_n + 1)
        # Loại bỏ chính game đầu vào
        recommended_indices = indices.flatten()[1:]
        recommended_games = data.iloc[recommended_indices][['TEN_GAME', 'DANH_GIA', 'NHA_PHAT_HANH']]
        return recommended_games
    except IndexError:
        return "Game không tồn tại trong dữ liệu!"

# --- Streamlit app ---
st.set_page_config(page_title="Game Recommendation", page_icon="🎮", layout="wide")

# CSS Styling
st.markdown("""
<style>
body {
    color: #E0E0E0; /* Light gray text */
}
.stApp {
    background-color: #262730; /* Dark background, similar to Steam */
    font-family: 'Arial', sans-serif;
}
.stTextInput input, .stSelectbox select {
    background-color: #3E3F47; /* Slightly lighter than main background */
    color: #E0E0E0;
    border: 2px solid #66c0f4; /* Highlight color for input fields */
    padding: 10px;
    border-radius: 5px;
}
.stSelectbox select option {
    background: #3E3F47;
    color: #E0E0E0;
}
.stButton button {
    background-color: #66c0f4; /* Steam blue accent color */
    color: #181A21;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}
.stButton button:hover {
    background-color: #3399ff; /* Lighter blue for hover effect */
}
.st-bb {
    overflow-y: auto;
    max-height: 250px;
}
.stDataFrame {
    background-color: #171a21;
    border-radius: 5px;
    color: #c8c8c8;
}
.st-bh, .st-el, .st-fm, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-g0, .st-g1, .st-g2, .st-g3, .st-g4, .st-g5, .st-g6, .st-g7, .st-g8, .st-g9, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg {
    color: #c8c8c8 !important;
}
th { /* Table headers */
    background-color: #373c47;
    color: #c8c8c8;
    text-align: left;
}
h1 {
    color: #66c0f4; /* Steam blue */
    text-shadow: none;
    font-family: "Motiva Sans", sans-serif;
    text-align: center;
    font-size: 36px;
    margin-bottom: 30px;
}
h3 {
    color: #66c0f4; /* Steam blue */
    font-family: "Motiva Sans", sans-serif;
    text-align: center;
    margin-top: 20px;
}
.css-10trblm { /* Text color for labels and placeholders */
    color: #E0E0E0;
}
.result-container {
    background-color: transparent;
    border-radius: 5px;
    padding: 10px;
    margin-top: 20px;
    color: #E0E0E0;
}
.css-12oz5g7, .css-10trblm  {
    color: #E0E0E0;
}
.recommendation-title {
    color: #66c0f4; /* Steam blue */
    font-size: 24px;
    margin-bottom: 10px;
}
.game-entry {
    background-color: #3E3F47;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
.game-title {
    font-size: 18px;
    color: #66c0f4;
}
.game-details {
    font-size: 14px;
    color: #c8c8c8;
}
.search-container {
    padding: 10px;
    border-radius: 5px;
}
.search-label {
    color: #66c0f4;
    margin-bottom: 5px;
}
.search-input {
    width: 100%;
    padding: 10px;
    border: 2px solid #66c0f4;
    border-radius: 5px;
    background-color: #3E3F47;
    color: #E0E0E0;
    margin-bottom: 10px;
}
.highlight {
    border: 2px solid #66c0f4;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1>🎮 Game Recommendation System 🎮</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #c8c8c8;'>Ứng dụng gợi ý game dựa trên game bạn đã chơi.</div>", unsafe_allow_html=True)

# Input and Results in columns
col1, col2 = st.columns([1, 2])

with col1:
    with st.container():
        st.markdown("<div class='search-label'>Nhập tên game:</div>", unsafe_allow_html=True)
        game_list = aggregated_data['TEN_GAME'].unique().tolist()
        game_name = st.selectbox("Select a game", game_list, index=game_list.index("Terraria") if "Terraria" in game_list else 0, key="game_select", label_visibility="collapsed")


        if st.button("Gợi ý"):
            message = st.info("Đang tìm kiếm...")
            recommendations = recommend_games_knn(game_name, aggregated_data, knn_model)
            message.empty()

            with col2:
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    st.markdown("<h3 class='recommendation-title'>Kết quả gợi ý:</h3>", unsafe_allow_html=True)
                    with st.container():
                        for _, row in recommendations.iterrows():
                            st.markdown(f"<div class='game-entry'><div class='game-title'>{row['TEN_GAME']}</div><div class='game-details'>Đánh giá: {row['DANH_GIA']}<br>Nhà phát hành: {row['NHA_PHAT_HANH']}</div></div>", unsafe_allow_html=True)

