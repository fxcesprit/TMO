# Amazon Books ML Project

Проект решает задачу бинарной классификации книг Amazon: Fiction / Non-Fiction.

## Состав файлов

- `amazon_books_ml_notebook.ipynb` — код для Jupyter Notebook.
- `app.py` — веб-приложение Streamlit.
- `requirements.txt` — зависимости Python.
- `Amazon_BestSelling_Books_500.xls` — датасет. Файл имеет расширение `.xls`, но читается как CSV.

## Запуск Jupyter Notebook

Создайте виртуальное окружение и установите зависимости:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    pip install notebook

Запустите Jupyter:

    jupyter notebook amazon_books_ml_notebook.ipynb

## Запуск Streamlit-приложения

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

После запуска Streamlit откроет приложение в браузере. В боковой панели можно менять гиперпараметры модели `RandomForestClassifier`: `n_estimators`, `max_depth`, `min_samples_split`, а также размер тестовой выборки. При изменении параметров модель переобучается.

## Docker build и run для Windows

Сборка образа:

    docker build -t amazon-books-ml .

Запуск контейнера:

    docker run --rm -p 8501:8501 amazon-books-ml

После запуска приложение будет доступно в браузере по адресу:

    http://localhost:8501
