import streamlit as st
import pandas as pd
import plotly.express as px
import socket
import requests
import subprocess

from surprise import Dataset, Reader, SVD, BaselineOnly, accuracy
from surprise.model_selection import train_test_split

st.set_page_config(page_title="Data Platform & RecSys", layout="wide")
st.title("Платформа хранения, обработки данных и рекомендательных систем")

# Вспомогательные функции
@st.cache_data
def load_raw_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies

def check_port(port):
    """Проверка доступности сетевого порта узла."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.3):
            return True
    except OSError:
        return False

def upload_to_seaweedfs(file_path):
    """Выделение тома с репликацией 001 и прямая запись файла в Volume-узел."""
    s = requests.Session()
    s.trust_env = False
    
    # Запрос тома с репликацией 001 (зеркалирование на 2 узла в одной стойке)
    assign = s.get("http://127.0.0.1:9333/dir/assign?replication=001", timeout=5).json()
    fid = assign["fid"]
    port = 8081 if "volume2" in assign.get("url", "") else 8080
    
    with open(file_path, "rb") as f:
        res = s.post(f"http://127.0.0.1:{port}/{fid}", files={"file": f}, timeout=10)
        
    if res.status_code in [200, 201]:
        return True, fid, port
    return False, None, None

def read_replicated_file(fid):
    """Отказоустойчивое чтение: опрашивает узлы 8080 и 8081 по очереди."""
    s = requests.Session()
    s.trust_env = False
    for port, name in [(8080, "Volume-01 (порт 8080)"), (8081, "Volume-02 (порт 8081)")]:
        try:
            res = s.get(f"http://127.0.0.1:{port}/{fid}", timeout=1)
            if res.status_code == 200:
                return res.content, name
        except Exception:
            continue
    return None, None

def get_cluster_topology():
    """Получение топологии кластера от управляющего узла Master."""
    try:
        s = requests.Session()
        s.trust_env = False
        res = s.get("http://127.0.0.1:9333/dir/status", timeout=5)
        if res.status_code == 200:
            return res.json().get("Topology", {})
    except Exception:
        pass
    return None

ratings_df, movies_df = load_raw_data()

# Вкладки проекта
tab_storage, tab_etl, tab_ml = st.tabs([
    "1. Распределенные СХД (SeaweedFS)", 
    "2. Обработка данных и ETL", 
    "3. Рекомендательная система (SVD)"
])


# 1. РАСПРЕДЕЛЕННЫЕ СИСТЕМЫ ХРАНЕНИЯ ДАННЫХ
with tab_storage:
    st.header("1. Распределенные системы хранения данных (Кластер SeaweedFS)")
    st.markdown("""
    **Архитектура:** Распределенное объектное хранилище SeaweedFS.  
    Реализовано разделение управляющего слоя метаданных (Master Node) и слоя физического хранения данных (Volume Nodes) с фактором репликации 2x.
    """)

    status_master = check_port(9333)
    status_vol1 = check_port(8080)
    status_vol2 = check_port(8081)
    status_filer = check_port(8888)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if status_master:
            st.success("**Master Node**\n- Порт: 9333\n- Роль: Координатор / Топология\n- Статус: ONLINE")
        else:
            st.error("**Master Node**\n- Порт: 9333\n- Статус: OFFLINE")

    with c2:
        if status_vol1:
            st.success("**Volume-01**\n- Порт: 8080\n- Роль: Storage Node #1\n- Статус: ONLINE")
        else:
            st.error("**Volume-01**\n- Порт: 8080\n- Статус: OFFLINE")

    with c3:
        if status_vol2:
            st.success("**Volume-02**\n- Порт: 8081\n- Роль: Storage Node #2\n- Статус: ONLINE")
        else:
            st.error("**Volume-02**\n- Порт: 8081\n- Статус: OFFLINE")

    with c4:
        if status_filer:
            st.success("**Filer / S3 Gateway**\n- Порт: 8888 / 8333\n- Роль: Точка входа / API\n- Статус: ONLINE")
        else:
            st.error("**Filer / S3**\n- Порт: 8888\n- Статус: OFFLINE")

    st.divider()

    st.subheader("Запись и верификация данных в распределенном хранилище")
    st.write("Данные сохраняются с фактором репликации 2x (зеркалируются на оба узла Volume-01 и Volume-02):")

    if st.button("Загрузить ratings.csv в хранилище (с репликацией)"):
        try:
            ok, fid, port = upload_to_seaweedfs("data/ratings.csv")
            if ok:
                st.session_state['last_fid'] = fid
                st.success(f"Объект успешно реплицирован на узлы кластера. File ID (FID): {fid}")
            else:
                st.error("Ошибка при сохранении объекта в узел хранения.")
        except Exception as e:
            st.error(f"Сбой связи с кластером: {e}")

    # Блок тестирования отказоустойчивости
    if 'last_fid' in st.session_state:
        fid = st.session_state['last_fid']
        st.write("---")
        st.subheader("Тестирование отказоустойчивости (High Availability)")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Имитировать аварию: Отключить Volume-01"):
                subprocess.run("docker stop seaweed_volume1", shell=True)
                st.warning("Отправлена команда на остановку контейнера seaweed_volume1.")
                st.rerun()

        with btn_col2:
            if st.button("Восстановить работу Volume-01"):
                subprocess.run("docker start seaweed_volume1", shell=True)
                st.success("Контейнер seaweed_volume1 запущен.")
                st.rerun()

        # Автоматическое чтение из выжившей реплики
        file_bytes, node_source = read_replicated_file(fid)
        if file_bytes:
            st.info(f"Файл доступен в распределенном хранилище. Активный источник чтения: **{node_source}**")
            st.download_button(
                label=f"Скачать файл (источник: {node_source})",
                data=file_bytes,
                file_name="verified_ratings_from_storage.csv",
                mime="text/csv"
            )
        else:
            st.error("Данные недоступны: все узлы хранения выведены из строя.")

    st.divider()

    with st.expander("Топология кластера (данные с Master Node)"):
        topology = get_cluster_topology()
        if topology:
            st.json(topology)
        else:
            st.info("Информация о топологии временно недоступна.")

    st.divider()
    st.subheader("Сырые события хранилища (Raw Storage Data)")
    st.write(f"Общее количество зарегистрированных событий: {len(ratings_df):,} строк")
    st.dataframe(ratings_df.head(5), use_container_width=True)

# 2. ТЕХНОЛОГИЯ ХРАНЕНИЯ И ОБРАБОТКИ ДАННЫХ (ETL)
with tab_etl:
    st.header("2. Технология хранения и обработки данных (ETL)")
    st.markdown("Пайплайн пакетной обработки данных: нормализация типов, объединение сущностей и формирование аналитической витрины (Data Mart).")

    @st.cache_data
    def run_etl(ratings, movies):
        df = ratings.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        merged = pd.merge(df, movies, on='movieId')
        
        movie_stats = merged.groupby(['movieId', 'title']).agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        
        popular_movies = movie_stats[movie_stats['rating_count'] >= 30].sort_values(by='rating_mean', ascending=False)
        return merged, popular_movies

    merged_df, popular_movies = run_etl(ratings_df, movies_df)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_dist = px.histogram(
            ratings_df, x="rating", nbins=10, 
            title="Распределение оценок пользователей (100k наблюдений)",
            labels={'rating': 'Оценка'}, color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_chart2:
        fig_top = px.bar(
            popular_movies.head(10), x="rating_mean", y="title", orientation='h',
            title="Топ-10 фильмов по среднему баллу (min 30 оценок)",
            labels={'rating_mean': 'Средний балл', 'title': 'Фильм'},
            color="rating_mean", color_continuous_scale="Blues"
        )
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Сформированная витрина данных (Data Mart)")
    st.dataframe(popular_movies.head(8), use_container_width=True)

# 3. ПРИКЛАДНЫЕ ЗАДАЧИ АНАЛИЗА ДАННЫХ (RECSYS)
with tab_ml:
    st.header("3. Прикладные задачи анализа данных: Рекомендательные системы")
    st.markdown("Матричная факторизация (SVD) на базе коллаборативной фильтрации против базового бейзлайна смещений.")

    @st.cache_resource
    def train_recsys(ratings):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Baseline
        base = BaselineOnly()
        base.fit(trainset)
        preds_base = base.test(testset)
        rmse_base = accuracy.rmse(preds_base, verbose=False)
        
        # SVD (50 латентных факторов)
        svd = SVD(n_factors=50, n_epochs=20, random_state=42)
        svd.fit(trainset)
        preds_svd = svd.test(testset)
        rmse_svd = accuracy.rmse(preds_svd, verbose=False)
        
        return svd, trainset, rmse_base, rmse_svd

    with st.spinner("Обучение моделей машинного обучения..."):
        model, trainset, rmse_base, rmse_svd = train_recsys(ratings_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline RMSE", f"{rmse_base:.4f}")
    m2.metric("SVD RMSE", f"{rmse_svd:.4f}", delta=f"-{(rmse_base - rmse_svd):.4f}")
    m3.metric("Размер обучающей выборки", f"{len(ratings_df):,} записей")

    st.divider()
    st.subheader("Генерация персональных рекомендаций")
    
    user_list = sorted(ratings_df['userId'].unique()[:50])
    selected_user = st.selectbox("Идентификатор пользователя (User ID):", user_list)
    
    if st.button("Сформировать рекомендации"):
        watched_movie_ids = set(ratings_df[ratings_df['userId'] == selected_user]['movieId'])
        all_movie_ids = set(movies_df['movieId'])
        unwatched = list(all_movie_ids - watched_movie_ids)
        
        predictions = [model.predict(selected_user, mid) for mid in unwatched]
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        top_5 = predictions[:5]
        
        rec_data = []
        for p in top_5:
            title = movies_df[movies_df['movieId'] == p.iid]['title'].values[0]
            genres = movies_df[movies_df['movieId'] == p.iid]['genres'].values[0]
            rec_data.append({
                "Фильм": title,
                "Жанры": genres,
                "Прогнозируемый балл": f"{p.est:.2f} / 5.0"
            })
            
        st.write(f"Результат персональной выдачи (Top-5) для пользователя {selected_user}:")
        st.table(pd.DataFrame(rec_data))
