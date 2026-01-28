"""
XGBoost Classifier implementation using PySpark
"""

import os
import logging
import argparse
from typing import Tuple, Dict, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    StringIndexer,
    Imputer
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

try:
    from xgboost.spark import SparkXGBClassifier
except ImportError as e:
    error_msg = str(e)
    if "sklearn" in error_msg.lower():
        raise ImportError(
            "XGBoost requires scikit-learn. Install: pip install scikit-learn>=1.0.0"
        ) from e
    else:
        raise ImportError(
            "XGBoost with Spark support is required. "
            "Install: pip install xgboost>=2.0.0 scikit-learn>=1.0.0"
        ) from e

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _check_java_version() -> Tuple[bool, str]:
    """Check if Java 17+ is available."""
    import shutil
    import subprocess
    import re
    
    java_path = shutil.which('java')
    if not java_path:
        java_home = os.environ.get('JAVA_HOME')
        if java_home:
            java_path = os.path.join(java_home, 'bin', 'java')
            if not os.path.exists(java_path):
                return False, "Java executable not found in JAVA_HOME"
        else:
            return False, "Java not found in PATH and JAVA_HOME is not set"
    
    try:
        result = subprocess.run(
            [java_path, '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5
        )
        version_match = re.search(r'version\s+"?(\d+)', result.stdout)
        if version_match:
            version = int(version_match.group(1))
            if version >= 17:
                return True, f"Java {version}"
            return False, f"Java {version} found, but Java 17+ is required"
        return False, "Could not parse Java version"
    except Exception as e:
        return False, f"Error checking Java version: {str(e)}"


def create_spark_session(
    app_name: str = "XGBoost_Classifier",
    driver_memory: str = "4g",
    executor_memory: str = "4g"
) -> SparkSession:
    """Create and return Spark session with XGBoost support."""
    java_ok, java_msg = _check_java_version()
    if not java_ok:
        raise RuntimeError(f"Java 17+ is required but not found: {java_msg}")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    
    logger.info(f"Spark session created: {app_name}")
    return spark


def load_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Load data from CSV file."""
    logger.info(f"Loading data from {data_path}")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(data_path)
    logger.info(f"Data loaded: {df.count()} rows, {len(df.columns)} columns")
    return df


def remove_outliers(df: DataFrame, outlier_thresholds: Optional[Dict[str, float]] = None) -> DataFrame:
    """Remove outliers from the dataset."""
    from pyspark.sql.types import DoubleType
    
    thresholds = outlier_thresholds or {
        'annual_inc': 250000,
        'dti': 50,
        'open_acc': 40,
        'total_acc': 80,
        'revol_util': 120,
        'revol_bal': 250000
    }
    
    df_numeric = df
    for col_name in thresholds.keys():
        if col_name in df.columns and dict(df.dtypes)[col_name] == 'string':
            df_numeric = df_numeric.withColumn(
                col_name,
                col(col_name).cast(DoubleType())
            )
    
    filter_condition = None
    for col_name, threshold in thresholds.items():
        if col_name in df_numeric.columns:
            condition = col(col_name).cast(DoubleType()) <= threshold
            if filter_condition is None:
                filter_condition = condition
            else:
                filter_condition = filter_condition & condition
    
    if filter_condition is not None:
        df_filtered = df_numeric.filter(filter_condition)
        logger.info(f"Outliers removed: {df.count()} -> {df_filtered.count()} rows")
        return df_filtered
    
    return df_numeric


def preprocess_data(
    df: DataFrame,
    target_col: str = 'loan_status'
) -> Tuple[DataFrame, list, PipelineModel]:
    """Preprocess data: handle missing values, encode categorical features, and prepare features vector."""
    numeric_cols = []
    categorical_cols = []
    
    for field in df.schema.fields:
        if field.name != target_col:
            dtype = field.dataType.typeName()
            if dtype in ['int', 'double', 'float', 'long', 'integer']:
                numeric_cols.append(field.name)
            elif dtype == 'string':
                categorical_cols.append(field.name)
    
    logger.info(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
    
    imputers = []
    imputed_cols = []
    for col_name in numeric_cols:
        imputer = Imputer(
            inputCols=[col_name],
            outputCols=[f"{col_name}_imputed"],
            strategy="mean"
        )
        imputers.append(imputer)
        imputed_cols.append(f"{col_name}_imputed")
    
    indexers = []
    indexed_cols = []
    for col_name in categorical_cols:
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=f"{col_name}_indexed",
            handleInvalid="keep"
        )
        indexers.append(indexer)
        indexed_cols.append(f"{col_name}_indexed")
    
    feature_cols = imputed_cols + indexed_cols
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    
    scaler = MinMaxScaler(
        inputCol="features_raw",
        outputCol="features",
        min=0.0,
        max=1.0
    )
    
    label_indexer = StringIndexer(
        inputCol=target_col,
        outputCol="label",
        handleInvalid="keep"
    )
    
    stages = imputers + indexers + [assembler, scaler, label_indexer]
    pipeline = Pipeline(stages=stages)
    
    logger.info("Fitting preprocessing pipeline...")
    pipeline_model = pipeline.fit(df)
    preprocessed_df = pipeline_model.transform(df)
    logger.info("Preprocessing completed")
    
    return preprocessed_df, feature_cols, pipeline_model


def train_xgboost_model(
    train_df: DataFrame,
    num_workers: int = 2,
    use_gpu: bool = False,
    max_depth: int = 6,
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: int = 1,
    reg_alpha: float = 0,
    reg_lambda: float = 1,
    scale_pos_weight: float = 1,
    random_state: int = 42
):
    """Train XGBoost classifier."""
    xgb_classifier = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        num_workers=num_workers,
        use_gpu=use_gpu,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state
    )
    
    logger.info("Training XGBoost model...")
    xgb_model = xgb_classifier.fit(train_df)
    logger.info("Training completed")
    
    return xgb_model


def evaluate_model(
    model,
    train_df: DataFrame,
    test_df: DataFrame,
    show_samples: bool = False
) -> Dict[str, float]:
    """Evaluate the trained model."""
    logger.info("Making predictions on training set...")
    train_predictions = model.transform(train_df)
    
    logger.info("Making predictions on test set...")
    test_predictions = model.transform(test_df)
    
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    train_roc_auc = binary_evaluator.evaluate(train_predictions)
    test_roc_auc = binary_evaluator.evaluate(test_predictions)
    
    multiclass_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    train_accuracy = multiclass_evaluator.evaluate(train_predictions)
    test_accuracy = multiclass_evaluator.evaluate(test_predictions)
    
    results = {
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    
    logger.info("Model Evaluation Results:")
    logger.info(f"  Train ROC-AUC: {train_roc_auc:.4f}")
    logger.info(f"  Test ROC-AUC: {test_roc_auc:.4f}")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    
    if show_samples:
        logger.info("Sample predictions (first 10 rows):")
        test_predictions.select("label", "prediction", "probability").show(10, truncate=False)
    
    return results


def main():
    """Main function to run the XGBoost classifier pipeline."""
    parser = argparse.ArgumentParser(description='XGBoost Classifier with PySpark')
    parser.add_argument('--data-path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--target-col', type=str, default='loan_status', help='Target column name')
    parser.add_argument('--train-split', type=float, default=0.67, help='Train/test split ratio')
    parser.add_argument('--model-path', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of XGBoost workers')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--max-depth', type=int, default=6, help='Max depth of trees')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
    parser.add_argument('--learning-rate', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--show-samples', action='store_true', help='Show sample predictions')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")
    
    spark = create_spark_session()
    
    try:
        df = load_data(spark, args.data_path)
        df_clean = remove_outliers(df)
        preprocessed_df, feature_cols, pipeline_model = preprocess_data(df_clean, target_col=args.target_col)
        
        train_ratio = args.train_split
        test_ratio = 1 - train_ratio
        train_df, test_df = preprocessed_df.randomSplit([train_ratio, test_ratio], seed=args.random_state)
        
        logger.info(f"Train set size: {train_df.count()}")
        logger.info(f"Test set size: {test_df.count()}")
        
        xgb_model = train_xgboost_model(
            train_df,
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.random_state
        )
        
        results = evaluate_model(xgb_model, train_df, test_df, show_samples=args.show_samples)
        
        if args.model_path:
            logger.info(f"Saving model to {args.model_path}")
            xgb_model.write().overwrite().save(args.model_path)
        
        logger.info("Pipeline completed successfully")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
