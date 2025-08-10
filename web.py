import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
import time

# Настройка страницы Streamlit
st.set_page_config(
    page_title="3D Кластеризация Pro",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Глобальная переменная для минимального радиуса
min_r = 1

def distance(x1, y1, x2, y2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def search_min(data):
    """Поиск локальных минимумов с использованием радиуса"""
    x = data['x'].tolist()
    y = data['y'].tolist()
    z = data['z'].tolist()
    local_min = []
    
    for i in range(len(x)):
        flag = True
        for j in range(len(x)):
            if i != j and distance(x[i], y[i], x[j], y[j]) <= min_r and z[j] < z[i]:
                flag = False
                break
        if flag:
            local_min.append([x[i], y[i], z[i]])
    
    local_min = pd.DataFrame(local_min)
    if not local_min.empty:
        local_min.columns = ['x', 'y', 'z']
    return local_min

def search_min2(data, k_neig):
    """Поиск локальных минимумов с использованием k ближайших соседей"""
    x = data['x'].tolist()
    y = data['y'].tolist()
    z = data['z'].tolist()
    local_min = []
    
    for i in range(len(x)):
        points = []
        for j in range(len(x)):
            if i != j:
                points.append([distance(x[i], y[i], x[j], y[j]), z[j]])
        
        if len(points) >= k_neig:
            points = sorted(points, key=lambda ax: ax[0])[:k_neig]
            zx = min(points, key=lambda ax: ax[1])[1]
            if zx > z[i]:
                local_min.append([x[i], y[i], z[i], sum(point[1] for point in points) / len(points)])
    
    local_min = pd.DataFrame(local_min)
    if not local_min.empty:
        local_min.columns = ['x', 'y', 'z', 'dz']
    return local_min

def search_min3(data, k_neig = 3):
    """Поиск локальных минимумов с вычислением средних значений для 3, 5 и 10 соседей"""
    x = data['x'].tolist()
    y = data['y'].tolist()
    z = data['z'].tolist()

    local_min = []

    for i in range(len(x)):
        points = []
        for j in range(len(x)):
            if i != j:
                points.append([x[j], y[j], z[j], distance(x[i], y[i], x[j], y[j])])
        
        if len(points) >= k_neig:
            points = sorted(points, key=lambda ax: ax[-1])
            zx = min(points[:k_neig], key=lambda ax: ax[2])[2]

            if zx > z[i]:
                # Вычисляем средние значения для разного количества соседей
                dz3 = sum(point[2] for point in points[:3]) / 3 if len(points) >= 3 else z[i]
                dz5 = sum(point[2] for point in points[:5]) / 5 if len(points) >= 5 else dz3
                dz10 = sum(point[2] for point in points[:10]) / 10 if len(points) >= 10 else dz5
                local_min.append([x[i], y[i], z[i], dz3, dz5, dz10])

    local_min = pd.DataFrame(local_min)
    if not local_min.empty:
        local_min.columns = ['x', 'y', 'z', 'dz3', 'dz5', 'dz10']
    return local_min

def perform_clustering(data, n_clusters=5):
    """Выполняет кластеризацию KMeans по трем признакам dz3, dz5, dz10"""
    required_features = ['dz3', 'dz5', 'dz10']
    if not all(col in data.columns for col in required_features):
        return np.zeros(len(data)), 1
    
    # Используем все три признака для кластеризации
    features = data[required_features].values
    
    # Нормализация признаков для лучшего качества кластеризации
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Используем только KMeans
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(features_scaled)
    
    return labels, n_clusters, features_scaled

def create_dbscan_plot(df, eps=0.27, min_samples=4, show_convex_hulls=True):
    """Создает 3D график с DBSCAN кластеризацией всех точек"""
    # Подготовка данных
    X = df[['x', 'y', 'z']].values
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN кластеризация
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Добавление меток кластеров
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Анализ результатов
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    # Создание графика
    fig = go.Figure()
    
    # Основная поверхность (опционально)
    fig.add_trace(go.Mesh3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        opacity=0.1,
        colorscale='Reds',
        name='Поверхность данных',
        showlegend=False,
        lighting=dict(ambient=0.4, diffuse=0.8)
    ))
    
    # Цветовая палитра для кластеров
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    # Добавление точек каждого кластера
    for cluster_id in sorted(set(clusters)):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        if cluster_id != -1:

            # Точки кластера
            if len(cluster_data) > 2:  # Показываем только кластеры с достаточным количеством точек
                color = colors[cluster_id % len(colors)]
                fig.add_trace(go.Scatter3d(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    z=cluster_data['z'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=0.8,
                        line=dict(width=1, color='black')
                    ),
                    name=f'Кластер {cluster_id} ({len(cluster_data)} точек)',
                    hovertemplate=f'<b>Кластер {cluster_id}</b><br>x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<extra></extra>'
                ))
                
                # Добавление выпуклых оболочек для кластеров
                if show_convex_hulls and len(cluster_data) >= 4:
                    try:
                        fig.add_trace(go.Mesh3d(
                            x=cluster_data['x'],
                            y=cluster_data['y'],
                            z=cluster_data['z'],
                            alphahull=5,  # Параметр для определения формы поверхности
                            opacity=0.3,
                            color=color,
                            name=f'Оболочка {cluster_id}',
                            showlegend=False,
                            lighting=dict(ambient=0.6, diffuse=0.4)
                        ))
                    except:
                        pass  # Иногда alphahull может не работать для некоторых конфигураций точек
    
    # Настройка макета
    fig.update_layout(
        title=f'DBSCAN Кластеризация (eps={eps}, min_samples={min_samples})<br>Найдено кластеров: {n_clusters}, Шумовых точек: {n_noise}',
        scene=dict(
            xaxis_title='X координата',
            yaxis_title='Y координата',
            zaxis_title='Z координата',
            aspectmode='data',  # Сохранение пропорций осей
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.6)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    
    # Добавление интерактивных кнопок
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
            
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.08,
                yanchor="top"
            )
        ]
    )
    
    return fig

def create_2d_plots(df, data, labels):
    """Создает 2D визуализации"""
    fig_2d = go.Figure()
    
    # Scatter plot с цветовым кодированием по Z
    fig_2d.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            color=df['z'],
            colorscale='Viridis',
            size=4,
            opacity=0.6,
            colorbar=dict(title="Z значения")
        ),
        name='Все точки',
        hovertemplate='x: %{x}<br>y: %{y}<br>z: %{text}<extra></extra>',
        text=df['z']
    ))
    
    # Локальные минимумы
    if not data.empty:
        fig_2d.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers',
            marker=dict(
                color=labels,
                colorscale='rainbow',
                size=10,
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Локальные минимумы',
            hovertemplate='Кластер: %{text}<br>x: %{x}<br>y: %{y}<extra></extra>',
            text=labels
        ))
    
    fig_2d.update_layout(
        title='2D проекция данных',
        xaxis_title='X',
        yaxis_title='Y',
        width=800,
        height=600
    )
    
    return fig_2d

def create_statistical_plots(data, labels):
    """Создает статистические графики"""
    if data.empty:
        return None, None, None
    
    # 1. Гистограммы для всех трех признаков
    fig_hist_all = go.Figure()
    
    # Добавляем гистограммы для каждого признака
    for i, feature in enumerate(['dz3', 'dz5', 'dz10']):
        fig_hist_all.add_trace(go.Histogram(
            x=data[feature],
            name=f'{feature}',
            opacity=0.7,
            nbinsx=20,
            offsetgroup=i
        ))
    
    fig_hist_all.update_layout(
        title='Распределение всех dz признаков',
        xaxis_title='Значения',
        yaxis_title='Частота',
        barmode='overlay'
    )
    
    # 2. Корреляционная матрица
    available_cols = ['x', 'y', 'z', 'dz3', 'dz5', 'dz10']
    correlation_data = data[available_cols].corr()
    fig_corr = px.imshow(
        correlation_data,
        text_auto=True,
        title='Корреляционная матрица всех признаков',
        color_continuous_scale='RdBu'
    )
    
    # 3. 3D scatter plot признаков с кластерами
    fig_3d_features = go.Figure()
    
    if len(set(labels)) > 1:
        fig_3d_features.add_trace(go.Scatter3d(
            x=data['dz3'],
            y=data['dz5'],
            z=data['dz10'],
            mode='markers',
            marker=dict(
                color=labels,
                colorscale='rainbow',
                size=8,
                opacity=0.8,
                colorbar=dict(title="Кластер")
            ),
            text=[f'Кластер: {label}' for label in labels],
            hovertemplate='<b>%{text}</b><br>dz3: %{x}<br>dz5: %{y}<br>dz10: %{z}<extra></extra>'
        ))
    else:
        fig_3d_features.add_trace(go.Scatter3d(
            x=data['dz3'],
            y=data['dz5'],
            z=data['dz10'],
            mode='markers',
            marker=dict(
                color='blue',
                size=8,
                opacity=0.8
            ),
            hovertemplate='dz3: %{x}<br>dz5: %{y}<br>dz10: %{z}<extra></extra>'
        ))
    
    fig_3d_features.update_layout(
        title='3D пространство признаков (dz3, dz5, dz10)',
        scene=dict(
            xaxis_title='dz3',
            yaxis_title='dz5',
            zaxis_title='dz10'
        )
    )
    
    return fig_hist_all, fig_corr, fig_3d_features

def analyze_clusters(data, labels):
    """Анализирует характеристики кластеров"""
    if data.empty:
        return pd.DataFrame()
    
    cluster_stats = []
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        if label == -1:  # DBSCAN noise points
            continue
        mask = labels == label
        cluster_data = data[mask]
        
        stats = {
            'Кластер': label,
            'Размер': len(cluster_data),
            'Средний X': cluster_data['x'].mean(),
            'Средний Y': cluster_data['y'].mean(),
            'Средний Z': cluster_data['z'].mean(),
            'Средний dz3': cluster_data['dz3'].mean(),
            'Средний dz5': cluster_data['dz5'].mean(),
            'Средний dz10': cluster_data['dz10'].mean(),
            'Std Z': cluster_data['z'].std(),
            'Std dz3': cluster_data['dz3'].std(),
            'Std dz5': cluster_data['dz5'].std(),
            'Std dz10': cluster_data['dz10'].std(),
            'Min Z': cluster_data['z'].min(),
            'Max Z': cluster_data['z'].max()
        }
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

def create_3d_plot(df, data, labels, surface_opacity=0.6, point_size=8):
    """Создает 3D график с поверхностью и кластерами"""
    fig = go.Figure()
    
    # 1. Поверхность (триангуляция) - опционально
    if st.session_state.get('show_surface', True):
        fig.add_trace(go.Mesh3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            opacity=surface_opacity,
            colorscale='Reds',
            name='Поверхность',
            showlegend=False,
            lighting=dict(ambient=0.4, diffuse=0.8)
        ))
    
    # Все точки как облако точек
    if st.session_state.get('show_all_points', False):
        fig.add_trace(go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['z'],
                colorscale='Viridis',
                opacity=0.3
            ),
            name='Все точки'
        ))
    
    # 2. Точки минимумов с кластерами
    if not data.empty and len(labels) > 0:
        # Создаем текст для hover с информацией о всех признаках
        hover_text = [
            f'Кластер: {label}<br>dz3: {dz3:.3f}<br>dz5: {dz5:.3f}<br>dz10: {dz10:.3f}' 
            for label, dz3, dz5, dz10 in zip(labels, data['dz3'], data['dz5'], data['dz10'])
        ]
        
        fig.add_trace(go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            mode='markers',
            marker=dict(
                size=point_size,
                color=labels,
                colorscale='rainbow',
                opacity=0.9,
                line=dict(width=1, color='black'),
                colorbar=dict(title="Кластер")
            ),
            name='Кластеры (по dz3, dz5, dz10)',
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
        ))
    
    # Настройка макета
    fig.update_layout(
        title='3D визуализация с многомерной кластеризацией',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.6)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        width=900,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig

def main():
    st.title("🧊 3D Визуализация с продвинутой кластеризацией")
    st.markdown("### Интерактивный анализ локальных минимумов и кластеризация данных")
    
    # Инициализация session state
    if 'show_surface' not in st.session_state:
        st.session_state.show_surface = True
    if 'show_all_points' not in st.session_state:
        st.session_state.show_all_points = False
    
    # Боковая панель с расширенными параметрами
    st.sidebar.header("🎛️ Параметры анализа")
    
    # Загрузка файла
    uploaded_file = st.sidebar.file_uploader("📁 Выберите CSV файл", type=['csv'])
    
    # Расширенные настройки визуализации
    st.sidebar.subheader("🎨 Визуализация")
    show_surface = st.sidebar.checkbox("Показать поверхность", value=True)
    show_all_points = st.sidebar.checkbox("Показать все точки", value=False)
    surface_opacity = st.sidebar.slider("Прозрачность поверхности", 0.0, 1.0, 0.6)
    point_size = st.sidebar.slider("Размер точек", 3, 15, 8)
    
    st.session_state.show_surface = show_surface
    st.session_state.show_all_points = show_all_points
    
    if uploaded_file is not None:
        try:
            # Progress bar для загрузки
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('🔄 Загрузка данных...')
            progress_bar.progress(10)
            
            # Загрузка данных
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(20)
            
            # Проверка наличия необходимых колонок
            required_columns = ['x', 'y', 'z']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ CSV файл должен содержать колонки: {required_columns}")
                st.stop()
            
            st.sidebar.success(f"✅ Данные загружены: {len(df)} точек")
            
            # Расширенные параметры алгоритма
            st.sidebar.subheader("🔧 Параметры анализа")
            k_neig = st.sidebar.slider("Количество ближайших соседей", min_value=3, max_value=20, value=3)
            
            # Параметры KMeans
            st.sidebar.subheader("🎯 Параметры кластеризации")
            n_clusters = st.sidebar.slider("Количество кластеров", min_value=2, max_value=15, value=5)
            
            st.sidebar.info("💡 Используется KMeans кластеризация по трем признакам: dz3, dz5, dz10")
            
            progress_bar.progress(30)
            
            # Создание вкладок для разных видов анализа
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Основная информация", 
                "🔍 Анализ минимумов", 
                "📈 3D Визуализация", 
                "📉 2D & Статистика",
                "💾 Экспорт данных"
            ])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Общее количество точек", len(df))
                    st.metric("Размерность данных", f"{df.shape[1]} колонок")
                
                with col2:
                    st.metric("Мин Z", f"{df['z'].min():.3f}")
                    st.metric("Макс Z", f"{df['z'].max():.3f}")
                
                with col3:
                    st.metric("Среднее Z", f"{df['z'].mean():.3f}")
                    st.metric("Стд Z", f"{df['z'].std():.3f}")
                
                st.subheader("🔍 Просмотр данных")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Первые 10 строк:**")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.write("**Статистическое описание:**")
                    st.dataframe(df.describe(), use_container_width=True)
            
            # Поиск локальных минимумов с использованием search_min3
            status_text.text('🔍 Поиск локальных минимумов...')
            progress_bar.progress(50)
            
            start_time = time.time()
            data = search_min3(df, k_neig)  # Используем search_min3
            search_time = time.time() - start_time
            
            progress_bar.progress(70)
            
            if data.empty:
                st.warning("⚠️ Локальные минимумы не найдены. Попробуйте изменить параметры.")
                st.stop()
            
            # Кластеризация
            status_text.text('🤖 Выполнение кластеризации...')
            progress_bar.progress(80)
            
            start_cluster_time = time.time()
            labels, n_clusters_actual, features_scaled = perform_clustering(data, n_clusters)
            cluster_time = time.time() - start_cluster_time
            
            progress_bar.progress(100)
            status_text.text('✅ Анализ завершен!')
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Отображение результатов во вкладках
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Найдено минимумов", len(data))
                with col2:
                    st.metric("Количество кластеров", n_clusters_actual)
                with col3:
                    st.metric("Время поиска", f"{search_time:.2f} сек")
                
                st.subheader("📋 Детали локальных минимумов")
                st.dataframe(data, use_container_width=True)
                
                # Показываем сравнение средних значений
                if not data.empty:
                    st.subheader("📊 Сравнение средних значений соседей")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Среднее dz3", f"{data['dz3'].mean():.3f}")
                    with col2:
                        st.metric("Среднее dz5", f"{data['dz5'].mean():.3f}")
                    with col3:
                        st.metric("Среднее dz10", f"{data['dz10'].mean():.3f}")
                
                # Анализ кластеров
                if n_clusters_actual > 0:
                    st.subheader("📊 Анализ кластеров")
                    cluster_analysis = analyze_clusters(data, labels)
                    st.dataframe(cluster_analysis, use_container_width=True)
            
            with tab3:
                st.subheader("🎮 Интерактивная 3D Визуализация")
                
                # Создание и отображение 3D графика
                fig_3d = create_3d_plot(df, data, labels, surface_opacity, point_size)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Дополнительные метрики
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Время кластеризации", f"{cluster_time:.2f} сек")
                with col2:
                    st.metric("Шумовые точки", 0)  # KMeans не создает шумовые точки
                with col3:
                    st.metric("Метод кластеризации", "KMeans")
                
                # Добавляем DBSCAN кластеризацию
                st.subheader("🔬 DBSCAN Кластеризация всех точек")
                st.markdown("*Анализ всего набора данных с помощью DBSCAN*")
                
                # Параметры DBSCAN
                col1, col2, col3 = st.columns(3)
                with col1:
                    eps = st.slider("DBSCAN eps (расстояние)", 0.1, 1.0, 0.27, 0.01)
                with col2:
                    min_samples = st.slider("DBSCAN min_samples", 2, 20, 4)
                with col3:
                    show_convex_hulls = st.checkbox("Показать выпуклые оболочки", value=True)
                
                if st.button("🚀 Запустить DBSCAN анализ"):
                    with st.spinner("Выполняется DBSCAN кластеризация..."):
                        fig_dbscan = create_dbscan_plot(df, eps, min_samples, show_convex_hulls)
                        st.plotly_chart(fig_dbscan, use_container_width=True)
                        
                        # Получаем результаты DBSCAN для отображения метрик
                        X = df[['x', 'y', 'z']].values
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters_dbscan = dbscan.fit_predict(X_scaled)
                        
                        n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
                        n_noise = list(clusters_dbscan).count(-1)
                        
                        # Метрики DBSCAN
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("DBSCAN Кластеры", n_clusters_dbscan)
                        with col2:
                            st.metric("Шумовые точки", n_noise)
                        with col3:
                            st.metric("% шума", f"{n_noise/len(df)*100:.1f}%")
                        with col4:
                            st.metric("Точек в кластерах", len(df) - n_noise)
                        
                        # Анализ DBSCAN кластеров
                        if n_clusters_dbscan > 0:
                            st.subheader("📊 Анализ DBSCAN кластеров")
                            df_with_dbscan = df.copy()
                            df_with_dbscan['cluster'] = clusters_dbscan
                            
                            dbscan_stats = []
                            for cluster_id in sorted(set(clusters_dbscan)):
                                if cluster_id != -1:  # Исключаем шумовые точки
                                    cluster_data = df_with_dbscan[df_with_dbscan['cluster'] == cluster_id]
                                    stats = {
                                        'Кластер': cluster_id,
                                        'Размер': len(cluster_data),
                                        'Средний X': cluster_data['x'].mean(),
                                        'Средний Y': cluster_data['y'].mean(),
                                        'Средний Z': cluster_data['z'].mean(),
                                        'Std X': cluster_data['x'].std(),
                                        'Std Y': cluster_data['y'].std(),
                                        'Std Z': cluster_data['z'].std(),
                                        'Min Z': cluster_data['z'].min(),
                                        'Max Z': cluster_data['z'].max()
                                    }
                                    dbscan_stats.append(stats)
                            
                            if dbscan_stats:
                                st.dataframe(pd.DataFrame(dbscan_stats), use_container_width=True)
            
            with tab4:
                st.subheader("📈 Дополнительные визуализации")
                
                # 2D проекция
                fig_2d = create_2d_plots(df, data, labels)
                st.plotly_chart(fig_2d, use_container_width=True)
                
                # Статистические графики
                col1, col2 = st.columns(2)
                
                if not data.empty:
                    fig_hist, fig_corr, fig_3d_features = create_statistical_plots(data, labels)
                    
                    with col1:
                        if fig_hist:
                            st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                
                # 3D пространство признаков
                if not data.empty:
                    st.subheader("🎯 3D пространство признаков")
                    if 'fig_3d_features' in locals():
                        st.plotly_chart(fig_3d_features, use_container_width=True)
                
                # Распределение по кластерам
                if len(labels) > 0:
                    st.subheader("📊 Распределение точек по кластерам")
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                
                # Дополнительный график для сравнения dz значений
                if not data.empty:
                    st.subheader("📈 Сравнение средних значений соседей")
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Box(y=data['dz3'], name='dz3 (3 соседа)'))
                    fig_compare.add_trace(go.Box(y=data['dz5'], name='dz5 (5 соседей)'))
                    fig_compare.add_trace(go.Box(y=data['dz10'], name='dz10 (10 соседей)'))
                    fig_compare.update_layout(
                        title='Распределение средних значений для разного количества соседей',
                        yaxis_title='Среднее значение z'
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
            
            with tab5:
                st.subheader("💾 Экспорт результатов")
                
                col1, col2, col3 = st.columns(3)
                
                # Подготовка данных для экспорта
                data_with_clusters = data.copy()
                data_with_clusters['cluster'] = labels
                
                with col1:
                    csv_data = data_with_clusters.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать результаты (CSV)",
                        data=csv_data,
                        file_name=f'local_minima_clusters_kmeans.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    # Создание отчета
                    cluster_analysis = analyze_clusters(data, labels)
                    report = f"""# Отчет по KMeans кластеризации локальных минимумов

## Параметры анализа:
- Метод поиска: search_min3
- Метод кластеризации: KMeans
- Признаки для кластеризации: dz3, dz5, dz10 (многомерная)
- Нормализация признаков: StandardScaler
- K соседей для поиска минимумов: {k_neig}
- Количество кластеров: {n_clusters}
- Общее количество точек: {len(df)}
- Найдено минимумов: {len(data)}
- Время поиска: {search_time:.2f} сек
- Время кластеризации: {cluster_time:.2f} сек

## Статистика средних значений:
- Среднее dz3: {data['dz3'].mean():.3f} ± {data['dz3'].std():.3f}
- Среднее dz5: {data['dz5'].mean():.3f} ± {data['dz5'].std():.3f}
- Среднее dz10: {data['dz10'].mean():.3f} ± {data['dz10'].std():.3f}

## Корреляции между признаками:
- Корреляция dz3-dz5: {data[['dz3', 'dz5']].corr().iloc[0,1]:.3f}
- Корреляция dz3-dz10: {data[['dz3', 'dz10']].corr().iloc[0,1]:.3f}
- Корреляция dz5-dz10: {data[['dz5', 'dz10']].corr().iloc[0,1]:.3f}

## Статистика по кластерам:
{cluster_analysis.to_string() if not cluster_analysis.empty else "Нет данных"}
                    """
                    
                    st.download_button(
                        label="📄 Скачать отчет (TXT)",
                        data=report,
                        file_name=f'analysis_report_kmeans.txt',
                        mime='text/plain'
                    )
                
                with col3:
                    if st.button("🔄 Сбросить анализ"):
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {str(e)}")
            st.exception(e)
    
    # Демонстрационные данные - когда файл НЕ загружен
    if uploaded_file is None:
        st.info("💡 Загрузите CSV файл для начала анализа или выберите демонстрационные данные")
        
        # Выбор типа демонстрационных данных
        demo_type = st.selectbox(
            "Выберите тип демонстрационных данных:",
            ["Синусоидальная поверхность", "Гауссовы холмы", "Случайная поверхность", "Седловая точка"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_points = st.slider("Количество точек", 100, 1000, 300)
        
        with col2:
            noise_level = st.slider("Уровень шума", 0.0, 0.5, 0.1)
        
        if st.button("🎲 Создать демонстрационные данные"):
            # Создание различных типов демонстрационных данных
            np.random.seed(42)
            x = np.random.uniform(-10, 10, n_points)
            y = np.random.uniform(-10, 10, n_points)
            
            if demo_type == "Синусоидальная поверхность":
                z = np.sin(np.sqrt(x**2 + y**2)) + np.random.normal(0, noise_level, n_points)
            elif demo_type == "Гауссовы холмы":
                z = np.exp(-(x**2 + y**2)/20) + 0.5*np.exp(-((x-3)**2 + (y-3)**2)/10) + np.random.normal(0, noise_level, n_points)
            elif demo_type == "Случайная поверхность":
                z = np.random.normal(0, 1, n_points) + 0.1*x*y + np.random.normal(0, noise_level, n_points)
            else:  # Седловая точка
                z = x**2 - y**2 + np.random.normal(0, noise_level, n_points)
            
            demo_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
            
            st.success("✅ Демонстрационные данные созданы!")
            st.write(f"Создано {len(demo_df)} точек типа '{demo_type}'")
            
            # Показываем превью данных
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Первые 5 строк:**")
                st.dataframe(demo_df.head())
            with col2:
                st.write("**Статистика:**")
                st.dataframe(demo_df.describe())

if __name__ == "__main__":
    main()