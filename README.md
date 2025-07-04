# cv-project
Командный итоговый проект "Поиск изображений по текстовому описанию с помощью общего пространства эмбеддингов"

## Команда проекта
- Дистлер Марина
- Конохова Екатерина
- Холичева Ангелина

## Структура проекта 

Проект организован следующим образом:

### **Jupyter-ноутбуки**  
- **`1-create_dataset.ipynb`** — парсинг сайтов для создания датасета  
- **`2-preprocessing.ipynb`** — предобработка данных  
- **`3-zero-shot.ipynb`** — использование  и тестирование предобученных моделей  
- **`4-finetuning.ipynb`** — дообучение моделей
- **`5-search-examples.ipynb`** — примеры работы итоговой модели  

### **Скрипты и данные**  
- **`dataset.py`** — класс датасета  
- **`utils.py`** — вспомогательные функции 
- **`semantic_test_queries.json`** — тестовые запросы для тестирования

### **Вспомогательные файлы**  
- **`.gitignore`** — игнорируемые git файлы 
- **`requirements.txt`** — список зависимостей Python  
- **`README.md`** — документация проекта (этот файл)  

### **Папки**  
- **`CV_site/`** — сайт для поиска дизайнов машинной вышивки по текстовому описанию (более подробно можно прочитать в CV_site/README.md)
- **`figures/`** — графики и визуализации (сохраненные изображения)  

