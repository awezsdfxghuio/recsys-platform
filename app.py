import streamlit as st
import pandas as pd
import plotly.express as px
import socket
import requests
import subprocess

st.set_page_config(page_title="Data Platform: РСХД & ТХОД", layout="wide")
st.title("Платформа хранения и пакетной обработки данных")

# Глобальная HTTP-сессия с отключенным системным прокси
HTTP = requests.Session()
HTTP.trust_env = False
JSON_HEADERS = {"Accept": "application/json"}

# ==========================================
# Вспомогательные функции
# ==========================================

@st.cache_data
def load_raw_data():
    """Кэшированная загрузка исходных датасетов."""
    return pd.read_csv("data/ratings.csv"), pd.read_csv("data/movies.csv")

def check_port(port):
    """Проверка доступности TCP-порта контейнера."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False

def upload_to_filer(file_bytes, filename):
    """Прямая отправка байтов из ОЗУ в SeaweedFS Filer с репликацией 001 (2x)."""
    try:
        url = f"http://127.0.0.1:8888/{filename}?replication=001"
        res = HTTP.post(url, files={"file": (filename, file_bytes)}, timeout=8)
        return res.status_code in [200, 201]
    except Exception:
        return False

def get_file_fid(filename):
    """Извлечение физического FID файла из метаданных Filer."""
    try:
        res = HTTP.get(f"http://127.0.0.1:8888/{filename}?metadata=true", headers=JSON_HEADERS, timeout=1)
        chunks = res.json().get("chunks", []) if res.status_code == 200 else []
        return chunks[0].get("file_id") if chunks else None
    except Exception:
        return None

def read_replicated_file(filename):
    """Отказоустойчивое чтение напрямую с Volume-узлов (8080 -> 8081) по FID."""
    fid = get_file_fid(filename)
    if fid:
        for port, name in [(8080, "Volume-01 (порт 8080)"), (8081, "Volume-02 (порт 8081)")]:
            try:
                res = HTTP.get(f"http://127.0.0.1:{port}/{fid}", timeout=0.5)
                if res.status_code == 200:
                    return res.content, name
            except Exception:
                continue
    return None, None

def get_filer_entries():
    """Список объектов файлового шлюза Filer."""
    try:
        res = HTTP.get("http://127.0.0.1:8888/?pretty=y", headers=JSON_HEADERS, timeout=2)
        return res.json().get("Entries", []) if res.status_code == 200 else []
    except Exception:
        return []

def get_cluster_topology():
    """Топология кластера от Master Node."""
    try:
        res = HTTP.get("http://127.0.0.1:9333/dir/status", timeout=3)
        return res.json().get("Topology", {}) if res.status_code == 200 else None
    except Exception:
        return None

ratings_df, movies_df = load_raw_data()

tab_storage, tab_etl = st.tabs([
    "1. Распределенные СХД (SeaweedFS)", 
    "2. Технологии хранения и обработки (ETL)"
])

# ==========================================
# 1. РАСПРЕДЕЛЕННЫЕ СИСТЕМЫ ХРАНЕНИЯ ДАННЫХ
# ==========================================
with tab_storage:
    st.header("1. Распределенные системы хранения данных (Кластер SeaweedFS)")
    st.markdown("Связка управляющего слоя (**Master :9333**), файлового шлюза (**Filer :8888**) и узлов хранения (**Volume :8080 / :8081**) с репликацией **2x (`001`)**.")

    # Отрисовка карточек мониторинга через компактный цикл
    nodes_info = [
        ("Master Node", 9333, "Координатор / Топология"),
        ("Volume-01", 8080, "Storage Node #1"),
        ("Volume-02", 8081, "Storage Node #2"),
        ("Filer Gateway", 8888, "Индекс файлов / API")
    ]
    for col, (title, port, role) in zip(st.columns(4), nodes_info):
        if check_port(port):
            col.success(f"**{title}**\n\n- Порт: {port}\n- Роль: {role}\n- Статус: ONLINE")
        else:
            col.error(f"**{title}**\n\n- Порт: {port}\n- Статус: OFFLINE")

    st.divider()

    # Управление отказоустойчивостью (всегда наверху)
    st.subheader("Управление отказоустойчивостью (Chaos Engineering)")
    c_stop, c_start = st.columns(2)
    if c_stop.button("Имитировать аварию: Отключить Volume-01", use_container_width=True):
        subprocess.run("docker stop seaweed_volume1", shell=True)
        st.rerun()
    if c_start.button("Восстановить работу Volume-01", use_container_width=True):
        subprocess.run("docker start seaweed_volume1", shell=True)
        st.rerun()

    st.divider()

    # Загрузка произвольного файла без создания временных файлов на диске
    st.subheader("Запись и верификация данных в распределенном хранилище")
    custom_file = st.file_uploader("Выберите файл для загрузки с репликацией 2x:", type=["png", "jpg", "jpeg", "txt", "csv", "pdf"])

    if custom_file and st.button(f"Отправить '{custom_file.name}' в SeaweedFS"):
        if upload_to_filer(custom_file.getvalue(), custom_file.name):
            st.session_state.update({'filename': custom_file.name, 'filetype': custom_file.type})
            st.success(f"Файл '{custom_file.name}' записан в кластер с репликацией 2x!")
        else:
            st.error("Ошибка записи: убедитесь, что все узлы Volume запущены.")

    # Встроенный проводник Filer (для работы через Cloudflare Tunnel)
    with st.expander("Проводник файловой системы SeaweedFS Filer (порт 8888)", expanded=True):
        entries = get_filer_entries()
        if entries:
            df_filer = pd.DataFrame([{
                "Путь в хранилище": e.get("FullPath"),
                "Физический Volume FID": e.get("chunks", [{}])[0].get("file_id", "Директория"),
                "Размер (байт)": e.get("FileSize", 0),
                "Время записи": e.get("Mtime")
            } for e in entries])
            st.dataframe(df_filer, use_container_width=True)
        else:
            st.info("Файловый шлюз пуст или недоступен.")

    # Проверка чтения из выжившей реплики
    if 'filename' in st.session_state:
        fname, ftype = st.session_state['filename'], st.session_state['filetype']
        st.write("---")
        st.subheader(f"Проверка доступности объекта: {fname}")

        file_bytes, node_source = read_replicated_file(fname)
        if file_bytes:
            st.info(f"Объект успешно прочитан! Активный источник: **{node_source}**")
            st.download_button(f"Скачать '{fname}' ({node_source})", file_bytes, f"restored_{fname}", ftype)
            if ftype.startswith("image/"):
                st.image(file_bytes, caption=f"Чтение напрямую из {node_source}", width=350)
        else:
            st.error("Данные недоступны: все узлы хранения выведены из строя.")

    st.divider()

    with st.expander("Топология кластера (данные с Master Node :9333)"):
        topo = get_cluster_topology()
        if topo:
            st.json(topo)
        else:
            st.info("Топология недоступна.")

    st.subheader("Сырые события хранилища (Raw Storage Data)")
    st.write(f"Всего записей: {len(ratings_df):,} строк")
    st.dataframe(ratings_df.head(5), use_container_width=True)

# ==========================================
# 2. ТЕХНОЛОГИЯ ХРАНЕНИЯ И ОБРАБОТКИ ДАННЫХ (ETL)
# ==========================================
with tab_etl:
    st.header("2. Технология хранения и обработки данных (ETL)")
    st.markdown("Пакетный пайплайн: нормализация временных меток, реляционное объединение (`JOIN`) и построение витрины данных (**Data Mart**).")

    @st.cache_data
    def run_etl(ratings, movies):
        df = ratings.assign(datetime=pd.to_datetime(ratings['timestamp'], unit='s'))
        merged = pd.merge(df, movies, on='movieId')
        stats = merged.groupby(['movieId', 'title']).agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        return stats[stats['rating_count'] >= 30].sort_values(by='rating_mean', ascending=False)

    popular_movies = run_etl(ratings_df, movies_df)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig_dist = px.histogram(
            ratings_df, x="rating", nbins=10, 
            title="Распределение оценок (100k наблюдений)",
            labels={'rating': 'Оценка'}, color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_chart2:
        fig_top = px.bar(
            popular_movies.head(10), x="rating_mean", y="title", orientation='h',
            title="Топ-10 фильмов по среднему баллу (min 30 оценок)",
            labels={'rating_mean': 'Средний балл', 'title': 'Фильм'},
            color="rating_mean", color_continuous_scale="Blues"
        ).update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Сформированная витрина данных (Data Mart)")
    st.dataframe(popular_movies.head(8), use_container_width=True)