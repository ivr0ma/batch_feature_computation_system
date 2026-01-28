# Batch Feature Computation System

Система для предсказания дефолта по кредитам с использованием XGBoost Classifier на PySpark.

## Описание

Проект реализует классификатор XGBoost для предсказания дефолта по кредитам на основе данных Lending Club. Реализация выполнена на PySpark для масштабируемой обработки больших объемов данных.

### Основные возможности

- Загрузка и предобработка данных из CSV
- Удаление выбросов
- Автоматическая обработка категориальных и числовых признаков
- Нормализация признаков (MinMaxScaler)
- Обучение XGBoost Classifier на PySpark
- Оценка модели (ROC-AUC, Accuracy)
- Распределенная обработка данных

## Требования

### Системные требования

- **Python**: 3.8+
- **Java**: JDK 17 или выше (обязательно для PySpark 3.5+)
- **ОС**: Linux, macOS, Windows (WSL)

### Python зависимости

Все зависимости указаны в `requirements.txt`:

- `pyspark>=3.5.0` - Apache Spark для Python
- `xgboost>=2.0.0` - XGBoost с поддержкой Spark
- `scikit-learn>=1.0.0` - Требуется для XGBoost
- `pyarrow>=15.0.0` - Требуется для передачи данных между Spark и XGBoost
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `jupyterlab>=4.0.0` - Для работы с ноутбуками

## Установка

### 1. Установка Java JDK 17+

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install openjdk-17-jdk
```

#### Проверка установки:
```bash
java -version  # Должно показать версию 17 или выше
```

#### Настройка переменных окружения:
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Для постоянной установки добавьте в ~/.bashrc:
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## Использование

### Запуск XGBoost Classifier на PySpark

Базовый запуск (обязательно указать путь к данным):

```bash
python xgboost_pyspark.py --data-path data/lending_club_loan_two.csv
```

### Параметры командной строки

```bash
python xgboost_pyspark.py \
    --data-path data/lending_club_loan_two.csv \
    --target-col loan_status \
    --train-split 0.67 \
    --model-path models/xgboost_model \
    --num-workers 2 \
    --max-depth 6 \
    --n-estimators 100 \
    --learning-rate 0.3 \
    --random-state 42 \
    --show-samples
```

#### Основные параметры:

- `--data-path` (обязательно) - Путь к CSV файлу с данными
- `--target-col` - Название целевой колонки (по умолчанию: `loan_status`)
- `--train-split` - Доля обучающей выборки (по умолчанию: `0.67`)
- `--model-path` - Путь для сохранения обученной модели (опционально)
- `--num-workers` - Количество worker процессов XGBoost (по умолчанию: `2`)
- `--use-gpu` - Использовать GPU для обучения (флаг)
- `--max-depth` - Максимальная глубина деревьев (по умолчанию: `6`)
- `--n-estimators` - Количество деревьев (по умолчанию: `100`)
- `--learning-rate` - Скорость обучения (по умолчанию: `0.3`)
- `--random-state` - Seed для воспроизводимости (по умолчанию: `42`)
- `--show-samples` - Показать примеры предсказаний (флаг)

### Примеры использования

Сохранить модель:
```bash
python xgboost_pyspark.py --data-path data/lending_club_loan_two.csv --model-path models/xgboost_model
```

Использовать GPU и показать примеры:
```bash
python xgboost_pyspark.py --data-path data/lending_club_loan_two.csv --use-gpu --show-samples
```

Настроить параметры модели:
```bash
python xgboost_pyspark.py \
    --data-path data/lending_club_loan_two.csv \
    --max-depth 8 \
    --n-estimators 200 \
    --learning-rate 0.1
```

### Сохранение и загрузка модели

Для сохранения обученной модели используйте параметр `--model-path`:

```bash
python xgboost_pyspark.py --data-path data/lending_club_loan_two.csv --model-path models/xgboost_model
```

Модель будет сохранена в указанной директории и может быть загружена для предсказаний:

```python
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifierModel

# Создание Spark сессии
spark = SparkSession.builder.appName("Load_XGBoost_Model").getOrCreate()

# Загрузка модели
loaded_model = SparkXGBClassifierModel.load("models/xgboost_model")

# Загрузка и предобработка данных для предсказаний
# (нужно применить ту же предобработку, что и при обучении)
from xgboost_pyspark import load_data, remove_outliers, preprocess_data

test_data = load_data(spark, "data/test_data.csv")
test_data = remove_outliers(test_data)
test_data, _, _ = preprocess_data(test_data, target_col='loan_status')

# Использование для предсказаний
predictions = loaded_model.transform(test_data)
predictions.select("label", "prediction", "probability").show(10)

spark.stop()
```

### Работа с Jupyter Notebook

Для работы с оригинальным ноутбуком:

```bash
jupyter lab
```

Откройте `lending-club-loan-defaulters-prediction.ipynb` в Jupyter Lab.

## Структура проекта

```
batch_feature_computation_system/
├── data/                                    # Данные
│   ├── lending_club_loan_two.csv           # Базовый датасет
│   ├── accepted_2007_to_2018Q4.csv        # Полный датасет (принятые заявки)
│   └── rejected_2007_to_2018Q4.csv        # Полный датасет (отклоненные заявки)
├── xgboost_pyspark.py                       # Реализация XGBoost на PySpark
├── lending-club-loan-defaulters-prediction.ipynb  # Оригинальный ноутбук
├── requirements.txt                         # Python зависимости
├── README.md                                # Этот файл
└── venv/                                    # Виртуальное окружение
```

## Конфигурация

### Настройка параметров модели

Все параметры модели настраиваются через аргументы командной строки (см. раздел "Использование"). Основные параметры XGBoost:

- `--num-workers` - Количество worker процессов (по умолчанию: 2)
- `--max-depth` - Максимальная глубина деревьев (по умолчанию: 6)
- `--n-estimators` - Количество деревьев (по умолчанию: 100)
- `--learning-rate` - Скорость обучения (по умолчанию: 0.3)
- `--use-gpu` - Использовать GPU для обучения

### Настройка памяти Spark

Параметры памяти Spark настраиваются в функции `create_spark_session()` в файле `xgboost_pyspark.py`:

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \      # Память драйвера
    .config("spark.executor.memory", "4g") \    # Память executor
    .config("spark.driver.maxResultSize", "2g")  # Максимальный размер результата
    .getOrCreate()
```

Для изменения этих параметров отредактируйте функцию `create_spark_session()` или передайте их при создании сессии.

## Результаты

После выполнения скрипта вы получите:

- **Train ROC-AUC**: Метрика на обучающей выборке
- **Test ROC-AUC**: Метрика на тестовой выборке
- **Train Accuracy**: Точность на обучающей выборке
- **Test Accuracy**: Точность на тестовой выборке
- **Примеры предсказаний**: Первые 10 строк с предсказаниями (если указан флаг `--show-samples`)

### Пример вывода:

```
2025-01-27 10:30:15 - __main__ - INFO - Loading data from data/lending_club_loan_two.csv
2025-01-27 10:30:20 - __main__ - INFO - Data loaded: 396030 rows, 27 columns
2025-01-27 10:30:25 - __main__ - INFO - Outliers removed: 396030 -> 385421 rows
2025-01-27 10:30:30 - __main__ - INFO - Found 15 numeric and 5 categorical columns
2025-01-27 10:30:35 - __main__ - INFO - Training XGBoost model...
2025-01-27 10:35:40 - __main__ - INFO - Training completed
2025-01-27 10:35:45 - __main__ - INFO - Model Evaluation Results:
2025-01-27 10:35:45 - __main__ - INFO -   Train ROC-AUC: 0.7939
2025-01-27 10:35:45 - __main__ - INFO -   Test ROC-AUC: 0.7282
2025-01-27 10:35:45 - __main__ - INFO -   Train Accuracy: 0.8225
2025-01-27 10:35:45 - __main__ - INFO -   Test Accuracy: 0.8060
2025-01-27 10:35:50 - __main__ - INFO - Pipeline completed successfully
```

## Особенности реализации

### Предобработка данных

1. **Удаление выбросов**: Фильтрация по пороговым значениям:
   - `annual_inc <= 250000`
   - `dti <= 50`
   - `open_acc <= 40`
   - `total_acc <= 80`
   - `revol_util <= 120`
   - `revol_bal <= 250000`

2. **Обработка пропусков**: Заполнение средними значениями для числовых признаков

3. **Кодирование категориальных признаков**: Использование `StringIndexer` для преобразования категорий в числовые индексы

4. **Нормализация**

## Примечания

- Скрипт автоматически проверяет версию Java перед запуском
- Разделение данных: 67% train / 33% test по умолчанию (настраивается через `--train-split`)
- Все параметры модели настраиваются через аргументы командной строки
- Модель можно сохранить для последующего использования (флаг `--model-path`)
- Логирование выполняется через стандартный модуль `logging`
- BrokenPipeError в конце выполнения - нормальное поведение при закрытии Spark сессии

## Дополнительные ресурсы

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [XGBoost Spark Integration](https://xgboost.readthedocs.io/en/latest/tutorials/spark_estimator.html)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
