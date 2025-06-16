import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

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


def clip_training_step(
    batch, 
    model, 
    model_name, 
    loss_img, 
    loss_txt, 
    optimizer, 
    scaler, 
    device
):
    optimizer.zero_grad()
        
    images, texts = batch
    images = images.to(device)
    texts = texts.to(device)

    if model_name == 'clip':
        logits_per_image, logits_per_text = model(images, texts)
    elif model_name == 'ruclip':
        logits_per_image, logits_per_text = model(texts, images)

    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

    if model_name == 'ruclip':
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    elif model_name == 'clip':
        total_loss.backward()
        optimizer.step()

    # ограничение logit_scale
    with torch.no_grad():
        model.logit_scale.data = torch.clamp(model.logit_scale.data, max=100)

    return total_loss.item()


def clip_evaluation_step(batch, model, loss_img, loss_txt, device):
    images, texts = batch
    images = images.to(device)
    texts = texts.to(device)

    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

    return total_loss.item(), mean_similarity_between_true(model, image_features, text_features)


def clip_training_loop(
    num_epochs, 
    model, 
    model_name, 
    train_dataloader,
    test_dataloader,
    loss_img, 
    loss_txt, 
    optimizer,
    lr_scheduler,
    scaler,
    device
):
    logs = {
        'train_step_loss': [],
        'train_epoch_loss': [],
        'val_epoch_loss': [],
        'val_epoch_similarity': []
    }

    for epoch in range(num_epochs):
        # training step
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train()
        epoch_loss = 0
        for batch in pbar:
            step_loss = clip_training_step(batch, model, model_name, 
                                           loss_img, loss_txt, optimizer, scaler, device)
            epoch_loss += step_loss
            logs['train_step_loss'].append(step_loss)
            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {step_loss:.4f}")
        epoch_loss /= len(train_dataloader)
        logs['train_epoch_loss'].append(epoch_loss)
        
        # evaluation step
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_similarity = 0
            for batch in test_dataloader:
                step_loss, similarity = clip_evaluation_step(batch, model, loss_img, loss_txt, device)
                epoch_loss += step_loss
                epoch_similarity += similarity
            
            epoch_loss /= len(test_dataloader)
            epoch_similarity /= len(test_dataloader)
            logs['val_epoch_loss'].append(epoch_loss)
            logs['val_epoch_similarity'].append(epoch_similarity)
        
        lr_scheduler.step(epoch_loss)
    
    return logs


def plot_training_logs(logs: dict, title: str = None, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(20,4))
    if title:
        fig.suptitle(title)
    
    axes[0].plot(logs['train_step_loss'])
    axes[0].set_xlabel('Step')
    axes[0].set_title('Training contrastive loss')
    axes[1].plot(logs['train_epoch_loss'], label='train')
    axes[1].plot(logs['val_epoch_loss'], label='val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_xticks(range(len(logs['train_epoch_loss'])))
    axes[1].set_title('Losses per epoch')
    axes[1].legend()
    axes[2].plot(logs['val_epoch_similarity'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_xticks(range(len(logs['val_epoch_similarity'])))
    axes[2].set_title('Validation cosine similarity')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    
    fig.show()


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


def mean_similarity_between_true(model, image_embeddings, text_embeddings, print_result=True):
    device = next(model.parameters()).device
    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)
    similarities = (model.logit_scale.exp() * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    mean_similarity = torch.diag(similarities).mean().item()
    if print_result:
        print(f'Среднее косинусное сходство эмбеддингов у правильных пар: {mean_similarity:.4f}')
    return mean_similarity