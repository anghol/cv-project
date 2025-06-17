import os
from flask import Flask, render_template, request, url_for, send_from_directory
import pandas as pd
import numpy as np
import json
import pickle
import torch
import ruclip
from PIL import Image
import faiss

app = Flask(__name__)

# --- Константы и конфигурация ---
CSV_METADATA_PATH = os.path.join(app.root_path, 'data', 'products.csv')
PKL_EMBEDDINGS_PATH = os.path.join(app.root_path, 'data', 'embeddings_ruclip_finetuned.pkl')
FAISS_INDEX_PATH = os.path.join(app.root_path, 'data', 'faiss_index.bin')
PRODUCT_DATA_JSON_PATH = os.path.join(app.root_path, 'data', 'products_combined_data.json')

# Глобальные переменные для хранения данных
COMBINED_PRODUCTS_DF = None
FAISS_INDEX = None
PRODUCT_ID_MAP = {}
RUCLIP_MODEL = None
RUCLIP_PREPROCESS = None
DEVICE = None
EMBEDDING_DIM = None

N_RESULTS = 9
SIMILARITY_THRESHOLD = 0.001

# --- Функция для получения эмбеддинга текста (для запроса) ---
def get_query_embedding(text_query: str, model, processor, model_name_str, device_str):
    assert model_name_str.lower() in ['clip', 'ruclip'], 'Unknown model name!'
    
    if model_name_str == 'clip':
        raise NotImplementedError("CLIP-specific tokenization not implemented in this version.")
    elif model_name_str == 'ruclip':
        text_processed = processor(text=[text_query], return_tensors='pt', padding=True, truncation=True)['input_ids']

    with torch.no_grad():
        text_embedding = model.encode_text(text_processed.to(device_str)).cpu()
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding.squeeze().numpy().astype('float32')

def load_model():
    """Инициализирует и загружает RuCLIP модель."""
    global RUCLIP_MODEL, RUCLIP_PREPROCESS, DEVICE, EMBEDDING_DIM

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {DEVICE}")

    print("Загрузка RuCLIP модели...")
    RUCLIP_MODEL, RUCLIP_PREPROCESS = ruclip.load('ruclip-vit-base-patch32-224', device=DEVICE)
    
    finetuned_weights_path = os.path.join(app.root_path, 'data', 'ruclip_finetuned.pt')
    if os.path.exists(finetuned_weights_path):
        RUCLIP_MODEL.load_state_dict(torch.load(finetuned_weights_path, map_location=DEVICE), strict=False)
        print(f"Загружены fine-tuned веса из {finetuned_weights_path}")
    else:
        print(f"Внимание: Fine-tuned веса не найдены по пути {finetuned_weights_path}. Используется базовая модель.")

    RUCLIP_MODEL.eval()
    
    with torch.no_grad():
        test_image_path = os.path.join(app.root_path, 'static', 'images', 'no_image.jpg')
        if not os.path.exists(test_image_path):
            Image.new('RGB', (224, 224), color = 'red').save(test_image_path)
        
        test_image = Image.open(test_image_path).convert('RGB')
        processed_image = RUCLIP_PREPROCESS(images=[test_image], return_tensors='pt')['pixel_values']
        test_embedding = RUCLIP_MODEL.encode_image(processed_image.to(DEVICE)).cpu()
        EMBEDDING_DIM = test_embedding.shape[1]
        os.remove(test_image_path)

    print(f"Размерность эмбеддингов изображений модели: {EMBEDDING_DIM}")

def prepare_data_and_index():
    """
    Загружает метаданные из CSV, эмбеддинги из PKL, объединяет их по URL
    и строит FAISS индекс на image_embeddings.
    """
    global COMBINED_PRODUCTS_DF, FAISS_INDEX, PRODUCT_ID_MAP, EMBEDDING_DIM

    # 1. Загрузка метаданных из CSV
    if not os.path.exists(CSV_METADATA_PATH):
        raise FileNotFoundError(f"Ошибка: CSV файл с метаданными не найден по пути {CSV_METADATA_PATH}.")
    
    print(f"Загрузка метаданных из {CSV_METADATA_PATH}...")
    
    specific_dtypes = {} # Упрощенный specific_dtypes
    for i in range(1, 29): # Подгоните диапазон, если колонок больше/меньше
        specific_dtypes[f'local_image_path_{i}'] = str
        specific_dtypes[f'image_url_{i}'] = str

    metadata_df = pd.read_csv(CSV_METADATA_PATH, dtype=specific_dtypes, low_memory=False)
    
    # Теперь итерируемся по всем колонкам DataFrame для заполнения NaN
    # Важно: 'url', 'id' и другие колонки, которые будут использоваться,
    # будут автоматически определены, но мы их все равно обработаем для NaN
    for col in metadata_df.columns:
        if col in specific_dtypes: # Если тип был явно указан
            metadata_df[col] = metadata_df[col].fillna('').astype(str)
        elif pd.api.types.is_object_dtype(metadata_df[col]): # Если Pandas определил как object (обычно строки)
            metadata_df[col] = metadata_df[col].fillna('').astype(str)
        # Для числовых колонок NaN останутся NaN, если их не указывать как str
    
    # Убедимся, что колонка 'url' присутствует и не пуста для объединения
    if 'url' not in metadata_df.columns:
        raise ValueError("В CSV файле отсутствует обязательная колонка 'url' для объединения.")

    metadata_df['url'] = metadata_df['url'].fillna('').astype(str) # Принудительно приводим url к str

    metadata_df = metadata_df[metadata_df['url'] != '']
    
    print(f"Загружено {len(metadata_df)} строк метаданных из CSV.")
    
    # 2. Загрузка эмбеддингов из PKL
    if not os.path.exists(PKL_EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Ошибка: PKL файл с эмбеддингами не найден по пути {PKL_EMBEDDINGS_PATH}.")
    
    print(f"Загрузка эмбеддингов из {PKL_EMBEDDINGS_PATH}...")
    with open(PKL_EMBEDDINGS_PATH, 'rb') as f:
        embeddings_data_from_pkl = pickle.load(f)

    required_pkl_keys = ['urls', 'image_embeddings']
    for key in required_pkl_keys:
        if key not in embeddings_data_from_pkl:
            raise ValueError(f"Ошибка: В PKL файле отсутствует обязательный ключ '{key}'.")
    
    embeddings_df = pd.DataFrame({
        'url': [str(x) for x in embeddings_data_from_pkl['urls']],
        'image_embedding': [np.array(e).astype('float32') for e in embeddings_data_from_pkl['image_embeddings'].tolist()]
    })
    
    embeddings_df = embeddings_df[embeddings_df['url'] != '']

    print(f"Загружено {len(embeddings_df)} эмбеддингов из PKL.")

    # 3. Объединение данных по 'url'
    COMBINED_PRODUCTS_DF = pd.merge(metadata_df, embeddings_df, on='url', how='inner')
    
    if COMBINED_PRODUCTS_DF.empty:
        raise ValueError("После объединения CSV и PKL данных по URL не осталось ни одного товара. Проверьте URL на совпадение.")

    print(f"Объединено {len(COMBINED_PRODUCTS_DF)} товаров.")

    # 4. Сохранение объединенных данных в JSON (для кэширования)
    os.makedirs(os.path.dirname(PRODUCT_DATA_JSON_PATH), exist_ok=True)
    # Сохраняем только те колонки, которые нам нужны в веб-приложении для отображения
    # Это предотвратит ошибки при сохранении произвольных типов данных
    columns_to_save = [
        'id', 'name', 'description', 'price', 'url', 'category',
        'Размер дизайна, мм', 'Количество стежков', 'Количество цветов', 'Форматы файлов',
        'image_embedding' # Для FAISS, если индекс не сохранен отдельно
    ]
    # Добавляем все колонки local_image_path_X и image_url_X
    for i in range(1, 29):
        col_lp = f'local_image_path_{i}'
        if col_lp in COMBINED_PRODUCTS_DF.columns and col_lp not in columns_to_save:
            columns_to_save.append(col_lp)
        col_iu = f'image_url_{i}'
        if col_iu in COMBINED_PRODUCTS_DF.columns and col_iu not in columns_to_save:
            columns_to_save.append(col_iu)

    # Отфильтровываем колонки, которых нет в DataFrame
    final_columns_to_save = [col for col in columns_to_save if col in COMBINED_PRODUCTS_DF.columns]
    COMBINED_PRODUCTS_DF[final_columns_to_save].to_json(PRODUCT_DATA_JSON_PATH, orient='records', indent=4)
    print(f"Объединенные данные товаров сохранены в {PRODUCT_DATA_JSON_PATH}")

    # 5. Построение FAISS индекса на image_embeddings
    image_embeddings_np = np.vstack(COMBINED_PRODUCTS_DF['image_embedding'].values).astype('float32')

    FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
    FAISS_INDEX.add(image_embeddings_np)
    
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
    print(f"FAISS индекс построен и сохранен в {FAISS_INDEX_PATH}")

    # 6. Создание PRODUCT_ID_MAP
    PRODUCT_ID_MAP = {str(row['id']): row.to_dict() for idx, row in COMBINED_PRODUCTS_DF.iterrows()}


def load_data():
    """Загружает модель, FAISS индекс и комбинированные данные товаров."""
    global COMBINED_PRODUCTS_DF, FAISS_INDEX, PRODUCT_ID_MAP, RUCLIP_MODEL, RUCLIP_PREPROCESS, DEVICE, EMBEDDING_DIM

    load_model()

    if (os.path.exists(FAISS_INDEX_PATH) and
            os.path.exists(PRODUCT_DATA_JSON_PATH) and
            os.path.exists(CSV_METADATA_PATH) and
            os.path.exists(PKL_EMBEDDINGS_PATH)):
        
        print("Загрузка FAISS индекса и комбинированных данных из файлов...")
        
        COMBINED_PRODUCTS_DF = pd.read_json(PRODUCT_DATA_JSON_PATH)

        # Принудительно приводим колонки к str после загрузки из JSON
        columns_to_ensure_str = [
            'id', 'name', 'description', 'price', 'url', 'category',
            'Размер дизайна, мм', 'Количество стежков', 'Количество цветов', 'Форматы файлов'
        ]
        for i in range(1, 29):
            columns_to_ensure_str.append(f'local_image_path_{i}')
            columns_to_ensure_str.append(f'image_url_{i}')

        for col in columns_to_ensure_str:
            if col in COMBINED_PRODUCTS_DF.columns:
                COMBINED_PRODUCTS_DF[col] = COMBINED_PRODUCTS_DF[col].fillna('').astype(str)
        
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
        
        COMBINED_PRODUCTS_DF['id'] = COMBINED_PRODUCTS_DF.index
        PRODUCT_ID_MAP = {str(row['id']): row.to_dict() for idx, row in COMBINED_PRODUCTS_DF.iterrows()}
        print(f"Загружено {len(COMBINED_PRODUCTS_DF)} товаров и FAISS индекс.")
    else:
        print("FAISS индекс, комбинированные данные, CSV или PKL файл не найдены. Подготавливаем данные и строим индекс...")
        prepare_data_and_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    search_results = []
    query_text = ""
    error_message = None

    if request.method == 'POST':
        query_text = request.form.get('query')
        if query_text:
            print(f"Поиск по запросу: '{query_text}'")
            try:
                query_embedding = get_query_embedding(query_text, RUCLIP_MODEL, RUCLIP_PREPROCESS, 'ruclip', DEVICE)
                query_embedding_2d = np.expand_dims(query_embedding, axis=0)

                if not isinstance(query_embedding_2d, np.ndarray) or query_embedding_2d.shape != (1, EMBEDDING_DIM):
                    error_message = f"Ошибка: Эмбеддинг запроса имеет неверную форму для FAISS. Ожидается (1, {EMBEDDING_DIM}), получено {query_embedding_2d.shape}. (Тип: {type(query_embedding_2d)})"
                    print(error_message)
                    return render_template('index.html', search_results=[], query_text=query_text, error_message=error_message)

            except Exception as e:
                error_message = f"Ошибка при получении эмбеддинга запроса: {e}"
                print(f"ERROR: {error_message}")
                return render_template('index.html', search_results=[], query_text=query_text, error_message=error_message)

            if FAISS_INDEX is None or FAISS_INDEX.ntotal == 0:
                error_message = "FAISS индекс не инициализирован или содержит 0 элементов!"
                print(f"ERROR: {error_message}")
                return render_template('index.html', search_results=[], query_text=query_text, error_message=error_message)

            try:
                distances, neighbor_indices = FAISS_INDEX.search(query_embedding_2d, N_RESULTS)
                
                neighbor_indices = neighbor_indices.squeeze()
                distances = distances.squeeze()
                
                print(f"FAISS found {len(neighbor_indices)} neighbors.")
            except Exception as e:
                error_message = f"Ошибка при получении соседей из FAISS: {e}"
                print(f"ERROR: {error_message}")
                return render_template('index.html', search_results=[], query_text=query_text, error_message=error_message)

            # Временный список для сбора данных, которые потом будут отсортированы
            collected_results = [] 

            for i, df_idx in enumerate(neighbor_indices):
                if df_idx not in COMBINED_PRODUCTS_DF.index:
                    print(f"Внимание: Индекс DataFrame {df_idx} из FAISS не найден в COMBINED_PRODUCTS_DF. Пропускаем.")
                    continue

                product_data = COMBINED_PRODUCTS_DF.loc[df_idx].to_dict()
                
                similarity = 1 - (distances[i] ** 2) / 2 # L2 dist to cosine sim
                if similarity < 0: similarity = 0
                if similarity > 1: similarity = 1

                # Собираем данные и схожесть в промежуточный список

                if similarity >= SIMILARITY_THRESHOLD: 
                    collected_results.append({
                        "product_data": product_data,
                        "similarity": similarity
                    })
                else:
                    # Опционально: Вывести, сколько результатов было отфильтровано
                    print(f"Отфильтрован результат с ID {product_data.get('id', 'Н/Д')} из-за низкой схожести: {similarity:.4f} < {SIMILARITY_THRESHOLD:.4f}")
            
            # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: СОРТИРОВКА! ---
            # Сортируем собранные результаты по убыванию схожести
            collected_results.sort(key=lambda x: x["similarity"], reverse=True)

            # Теперь формируем конечный search_results из отсортированного списка
            for item in collected_results:
                product_data = item["product_data"]
                similarity = item["similarity"]

                product_result = {
                    "id": str(product_data.get('id', '')),
                    "name": product_data.get('name', f"Товар {product_data.get('id', 'Н/Д')}"),
                    "price": product_data.get('price', 'Н/Д'),
                    "url": product_data.get('url', '#'),
                    "similarity": f"{similarity:.4f}",
                    "image_paths": []
                }

                found_images = False
                for col_name in [c for c in COMBINED_PRODUCTS_DF.columns if 'local_image_path' in c]:
                    relative_path_from_csv = product_data.get(col_name)
                    if relative_path_from_csv:
                        corrected_relative_path = relative_path_from_csv 

                        full_img_path = os.path.join(app.root_path, 'static', corrected_relative_path)
                        if os.path.exists(full_img_path):
                            product_result['image_paths'].append(corrected_relative_path)
                            found_images = True
                        else:
                            print(f"Внимание: Изображение не найдено по пути: {full_img_path} для товара ID {product_result['id']} (колонка: {col_name})")
                
                if not found_images:
                    product_result['image_paths'].append(os.path.join('images', 'no_image.jpg'))

                search_results.append(product_result)
            print(f"Final search_results count: {len(search_results)}")

    return render_template('index.html', search_results=search_results, query_text=query_text, error_message=error_message)

@app.route('/product/<product_id>')
def product_detail(product_id):
    product_data_dict = PRODUCT_ID_MAP.get(product_id)
    
    if product_data_dict:
        image_paths = []
        
        for col_name in [c for c in COMBINED_PRODUCTS_DF.columns if 'local_image_path' in c]:
            relative_path_from_csv = product_data_dict.get(col_name)
            if relative_path_from_csv:
                corrected_relative_path = relative_path_from_csv
                full_img_path = os.path.join(app.root_path, 'static', corrected_relative_path)
                if os.path.exists(full_img_path):
                    image_paths.append(corrected_relative_path)
                else:
                    print(f"Внимание: Изображение не найдено по пути: {full_img_path} для товара ID {product_id} на странице деталей (колонка: {col_name}).")

        if not image_paths:
            print(f"Ошибка: Ни одного изображения не найдено для товара ID {product_id}. Отображаем заглушку.")
            image_paths.append(os.path.join('images', 'no_image.jpg'))

        # Теперь мы получаем данные динамически, поэтому используем .get() везде с дефолтными значениями
        product = {
            "id": str(product_data_dict.get('id', '')),
            "name": product_data_dict.get('name', f"Товар {product_data_dict.get('id', 'Н/Д')}"),
            "description": product_data_dict.get('description', 'Нет описания').replace('\r\n', '<br>').replace('\n', '<br>').replace('\r', '<br>'),
            "price": product_data_dict.get('price', 'Н/Д'),
            "url": product_data_dict.get('url', '#'),
            "category": product_data_dict.get('category', 'Н/Д'),
            "design_size": product_data_dict.get('Размер дизайна, мм', 'Н/Д'),
            "stitch_count": product_data_dict.get('Количество стежков', 'Н/Д'),
            "color_count": product_data_dict.get('Количество цветов', 'Н/Д'),
            "file_formats": product_data_dict.get('Форматы файлов', 'Н/Д'),
            "image_paths": image_paths
        }
        return render_template('product.html', product=product)
    return "Товар не найден", 404

@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    load_data()
    app.run()