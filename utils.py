import re
import pandas as pd
from tqdm import tqdm
from PIL import Image

import clip
import torch


def clean_embroteka_text(row: str):
    """ Очистка столбцов name и category для данных embroteka. """

    row = row.lower().strip()
    for symb in ['[', ']', ',', '\'', '"', '/', '|']:
        row = row.replace(symb, '')
    row = row.replace(' и ', ' ')
    row = row.replace('для детей', 'детские')
    return row.strip()


def clean_royal_text(row: str):
    """ Очистка столбцов name и category для данных royal present. """

    row = row.lower().strip()
    for symb in ['[', ']', ',', '\'', '"', '/', '|']:
        row = row.replace(symb, '')
    
    row = row.replace('дизайн машинной вышивки', '')
    row = row.replace('дизайн для машинной вышивки', '')
    row = re.sub(r'\s*дизайн\w*\s*', ' ', row)
    row = re.sub(r'\s*вышивк\w*\s*', ' ', row)
    row = re.sub(r'\s*машинной\w*\s*', ' ', row)
    row = re.sub(r'-\s*\d+\s*размер\w*\s*', ' ', row)
    row = row.replace(' - ', ' ')
    return row.strip()


def get_full_embeddings(
    df: pd.DataFrame, 
    model, 
    processor, 
    model_name: str,
    device, 
    batch_size: int = 128
):
    assert model_name.lower() in ['clip', 'ruclip'], 'Unknown model name!'

    total_image_embeddings, total_text_embeddings = [], []
    n_batches = (df.shape[0] // batch_size) + int(df.shape[0] % batch_size > 0)
    for i in tqdm(range(n_batches)):
        # получаем кусок датафрейма
        if i == n_batches - 1:
            df_slice = df.iloc[i*batch_size : ].copy()
        else:
            df_slice = df.iloc[i*batch_size : (i+1)*batch_size].copy()

        # батч предобработанных картинок-тензоров и батч токенизированных текстов
        tensor_images, tensor_texts = [], []
        for image_path, text in zip(df_slice['local_image_path'], df_slice['text']):
            if model_name == 'clip':
                tensor_images.append(processor(Image.open(image_path)).unsqueeze(0))
                tensor_texts.append(clip.tokenize([text], truncate=True))
            elif model_name == 'ruclip':
                tensor_images.append(processor(images=[Image.open(image_path)], return_tensors='pt')['pixel_values'])
                tensor_texts.append(processor(text=[text], return_tensors='pt', padding=True, truncation=True)['input_ids'])
        
        tensor_images = torch.cat(tensor_images).to(device)
        tensor_texts = torch.cat(tensor_texts).to(device)

        # получение эмбеддингов картинок и текста
        with torch.no_grad():
            image_embeddings = model.encode_image(tensor_images).cpu()
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            total_image_embeddings.append(image_embeddings)

            text_embeddings = model.encode_text(tensor_texts).cpu()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            total_text_embeddings.append(text_embeddings)
        
    total_image_embeddings = torch.cat(total_image_embeddings)
    total_text_embeddings = torch.cat(total_text_embeddings)

    return total_image_embeddings, total_text_embeddings


def get_text_embedding(text: str, model, processor, model_name, device):
    assert model_name.lower() in ['clip', 'ruclip'], 'Unknown model name!'
    
    if model_name == 'clip':
        text = clip.tokenize([text], truncate=True)
    elif model_name == 'ruclip':
        text = processor(text=[text], return_tensors='pt', padding=True, truncation=True)['input_ids']

    with torch.no_grad():
        text_embedding = model.encode_text(text.to(device)).cpu()
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding


def precision_at_k(relevant_dict, predicted_dict, k=5):
    print('----- Precision@5 -----')
    metric_values = []
    for query in relevant_dict:
        relevant_ids = set(relevant_dict[query]['ids'])
        predicted_ids = set(predicted_dict[query]['ids'])
        tp = relevant_ids & predicted_ids
        precision = len(tp) / k
        
        print(f'Запрос "{query}":  {precision:.2f}')
        metric_values.append(precision)
    
    print(f'Среднее значение по всем запросам: {np.mean(metric_values)}')


def recall_at_k(relevant_dict, predicted_dict, k=5):
    print('----- Recall@5 -----')
    metric_values = []
    for query in relevant_dict:
        relevant_ids = set(relevant_dict[query]['ids'])
        predicted_ids = set(predicted_dict[query]['ids'][:k])
        tp = relevant_ids & predicted_ids
        recall = len(tp) / len(relevant_ids)
        
        print(f'Запрос "{query}":  {recall:.2f}')
        metric_values.append(recall)
    
    print(f'Среднее значение по всем запросам: {np.mean(metric_values)}')


def mean_similarity_between_true(image_embeddings, text_embeddings):
    similarities = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    mean_similarity = torch.diag(similarities).mean().item()
    print(f'Среднее косинусное сходство эмбеддингов у правильных пар: {mean_similarity}')