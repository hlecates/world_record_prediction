# Aqua Analytics - National Level

A machine learning project that predicts swimming race outcomes and performance relative to records using comprehensive feature engineering on USA Swimming meet data.

## Project Overview

This project analyzes swimming competition data to predict:
1. **Winner's time relative to world records** (regression)
2. **Winner's time relative to American records** (regression) 
3. **Winner's time relative to US Open records** (regression)
4. **Whether the top seed wins** (classification)

The models achieve strong predictive performance with R² scores up to 0.958 for record residual predictions and incorporate sophisticated features like field competitiveness, psychological pressure indicators, and record proximity metrics.

## Key Results

### Regression Performance (Predicting Time Residuals)
- **US Open Record Residual**: R² = 0.958, RMSE = 2.74s (Gradient Boosting)
- **World Record Residual**: R² = 0.943, RMSE = 3.22s (Gradient Boosting)
- **American Record Residual**: R² = 0.938, RMSE = 3.41s (Gradient Boosting)

### Classification Performance (Top Seed Win Prediction)
- **Best Model**: Random Forest with 58.2% accuracy, 62.5% AUC
- **Challenge**: Inherent unpredictability in sports outcomes

### Most Important Features
1. **Seed time statistics** (mean, median) - dominant predictors
2. **Record proximity** - how close swimmers are to existing records
3. **Field competitiveness** - psychological pressure indicators
4. **Meet characteristics** - total record holders present

## Project Structure

```
national/
├── src/
│   ├── pipeline.py       # Data collection and parsing
│   ├── features.py       # Feature engineering
│   ├── modeling.py       # Model training and evaluation
│   ├── config.py         # Configuration settings
│   └── utils.py          # Utility functions
├── data/
│   ├── raw/              # Original PDF meet results
│   ├── processed/
│   │   ├── parsed/       # Structured data from PDFs
│   │   ├── clean/        # Cleaned event data
│   │   └── features/     # Engineered features
├── output/
│   ├── models/           # Trained models and preprocessors
│   └── plots/            # Performance visualizations
└── notebooks/            # Analysis notebooks
```

## Pipeline Workflow

### 1. Data Collection (`pipeline.py`)
- **Web scraping**: Automatically downloads PDF meet results from USA Swimming
- **PDF parsing**: Extracts structured data from competition PDFs using regex patterns
- **Data cleaning**: Removes incomplete events and standardizes formats

### 2. Feature Engineering (`features.py`)
The feature engineering creates **60+ sophisticated features** across multiple categories:

#### Field Depth & Competitiveness
- Basic statistics (mean, median, std of seed times)
- Distribution metrics (skewness, kurtosis, IQR)
- Competitiveness measures (Herfindahl-Hirschman Index)
- Gap analysis between seed times

#### Record Proximity Features
- Distance from top seed to world/American/US Open records
- Count of swimmers within percentage thresholds of records
- Clustering analysis around top performers

#### Psychological Pressure Indicators
- **Pressure Index**: Gap between 1st and 2nd seed (dominance measure)
- **Dark Horse Potential**: Strength of mid-field relative to top 2
- **Competitive Bandwidth**: Time spread of middle 50% of field

#### Participant Analysis
- Age demographics and range
- Record holder participation tracking
- Meet-level aggregated statistics

### 3. Model Training (`modeling.py`)
- **Preprocessing**: Feature selection, scaling, categorical encoding
- **Model comparison**: Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
- **Hyperparameter tuning**: Grid search with cross-validation
- **Evaluation**: Comprehensive metrics for both regression and classification

## Model Selection & Evaluation Strategy

#### Regression Models (for predicting time residuals):

**Ridge Regression** serves as our baseline linear model with built-in regularization. Since many swimming features are naturally correlated (field depth metrics, seed time statistics), Ridge handles this multicollinearity by adding an L2 penalty that shrinks coefficients without eliminating them entirely. This prevents overfitting while maintaining interpretability of the linear relationships.

**Lasso Regression** takes a different approach by performing automatic feature selection through its L1 penalty, which drives less important coefficients to exactly zero. This creates sparse models that identify the most critical features among our 60+ engineered variables, helping us understand which aspects of competition dynamics matter most for predicting race outcomes.

**Elastic Net** combines the strengths of both Ridge and Lasso by using a mixing parameter to balance L1 and L2 penalties. This hybrid approach handles correlated features while still performing feature selection, making it particularly suited for our swimming dataset where there are many potentially relevant but related features.

**Random Forest Regressor** captures complex non-linear relationships and feature interactions without making assumptions about data distribution. It builds an ensemble of decision trees using bootstrap sampling and random feature selection, providing robust predictions and natural feature importance rankings that help us understand competitive dynamics.

**Gradient Boosting Regressor** builds models iteratively, with each new model correcting the errors of previous ones. This sequential error correction often achieves the highest predictive accuracy on structured data like ours, where subtle patterns in competitive dynamics can significantly impact performance.

#### Classification Models (for top seed win prediction):

**Logistic Regression** provides an interpretable baseline that models the probability of the top seed winning using a logistic function. It offers clear coefficient interpretations and probability estimates, helping us understand which factors linearly influence the likelihood of upsets in swimming competitions.

**Random Forest Classifier** uses majority voting from an ensemble of decision trees to make robust predictions. This approach handles class imbalance well and provides stable performance across different competition conditions, making it particularly valuable for the inherently noisy task of predicting sports outcomes.

**Gradient Boosting Classifier** applies the same sequential error correction approach as its regression counterpart, often achieving the best classification performance on structured data by learning complex patterns that single models might miss.

### Evaluation Metrics Explained

#### For Regression Tasks (Time Residual Prediction):

**R² Score (Coefficient of Determination)** represents the proportion of variance in the target variable that our model explains, ranging from 0 to 1 where higher values indicate better performance. An R² of 0.95 tells us the model explains 95% of the variance in swimming times relative to records. Our results of 0.94-0.96 demonstrate excellent explanatory power, indicating the models capture most of the systematic factors influencing race outcomes.

**RMSE (Root Mean Squared Error)** measures the average prediction error in the same units as our target variable (seconds), penalizing larger errors more heavily than smaller ones. This metric is particularly meaningful for swimming predictions because it directly tells us how far off our time predictions typically are. Our results of approximately 3 seconds average error represent quite good accuracy for predicting competitive swimming performance.

**MAE (Mean Absolute Error)** provides the average absolute prediction error, offering a more robust measure against outliers compared to RMSE while being easier to interpret in practical terms.

#### For Classification Task (Top Seed Win):

**Accuracy** simply measures the percentage of correct predictions, providing an intuitive baseline metric. Our 58% accuracy is reasonable given the inherent unpredictability in sports, where upsets are common and many factors beyond our features influence outcomes.

**AUC (Area Under ROC Curve)** evaluates the model's ability to distinguish between classes across all possible decision thresholds, ranging from 0.5 (random guessing) to 1.0 (perfect classification). This metric is more robust than accuracy when dealing with imbalanced classes. Our result of 0.625 shows meaningful predictive ability above random chance, indicating the model captures real patterns in competitive dynamics.

**F1 Score** represents the harmonic mean of precision and recall, providing a balanced measure that accounts for both false positives and false negatives. This metric is particularly valuable since both missing true positive predictions and having false alarms is weighed.

**Precision** measures the reliability of positive predictions by calculating what percentage of predicted wins were actually wins, while **Recall** measures completeness by determining what percentage of actual wins are successfully predicted. Together, these metrics provide insight into different aspects of model performance depending on the specific use case.

### Why These Metrics Matter

1. **R² and RMSE together**: R² shows overall model quality, RMSE shows practical prediction accuracy
2. **Multiple classification metrics**: Each captures different aspects of performance important for different use cases
3. **Cross-validation**: All metrics computed on held-out test data to ensure real-world performance
4. **Overfitting detection**: Comparing train vs test performance to identify models that memorize vs generalize

### Model Selection Results

**For Regression**: Gradient Boosting won on all tasks due to its ability to capture complex non-linear relationships in swimming performance data.

**For Classification**: Random Forest performed best, likely due to its robustness with the inherent noise in predicting sports outcomes.

## Technical Implementation

### Model Architecture
- **Feature selection**: SelectKBest with univariate statistical tests
- **Scaling**: StandardScaler for numerical stability
- **Cross-validation**: Stratified K-Fold for classification, K-Fold for regression
- **Ensemble methods**: Random Forest and Gradient Boosting for best performance

## Model Performance Analysis

### Strengths
1. **Excellent regression performance** - Models explain 94-96% of variance in record residuals
2. **Robust feature engineering** - Captures complex competitive dynamics
3. **Comprehensive evaluation** - Multiple metrics and cross-validation
4. **Practical insights** - Feature importance reveals key competitive factors

## Usage

### Training New Models
```bash
# Run complete pipeline
python src/pipeline.py

# Generate features
python src/features.py

# Train models
python src/modeling.py
```

### Using Existing Data
```bash
# Parse existing PDFs
python src/pipeline.py --parse

# Clean existing data
python src/pipeline.py --clean
```

## Key Insights

1. **Seed time statistics dominate predictions** - The quality of the field (mean/median seed times) is the strongest predictor of race outcomes
2. **Record proximity matters** - How close swimmers are to existing records significantly impacts performance
3. **Psychological factors are measurable** - Pressure indices and competitive dynamics show predictive power
4. **Meet characteristics influence outcomes** - The presence of record holders and overall meet competitiveness affects results

## Limitations

1. #### Inherent Sports Unpredictability

Swimming, like all competitive sports, involves significant human performance variability that cannot be fully captured by historical data alone. Athletes can deliver breakthrough performances or experience unexpected poor showings regardless of their seeding or past performance. The psychological and physiological state of an athlete on race day represents a largely unmeasurable variable that can dramatically influence outcomes.

Numerous intangible factors play crucial roles in determining race results. Pool conditions such as water temperature, lane assignments, pool depth, and crowd noise can affect performance in ways that are difficult to quantify. The swimmer's psychology on race day—including confidence levels, pre-race nerves, motivation, and rivalry dynamics—represents another layer of complexity that resists numerical modeling. Additionally, an athlete's training state, including taper quality, injury status, and seasonal preparation cycles, varies significantly and is rarely fully documented in publicly available data.

2. #### Data Availability and Quality Constraints

The models suffer from significant gaps in critical information that would improve prediction accuracy. Training data such as weekly training volumes, training intensities, and periodization phases remain largely inaccessible, yet these factors fundamentally influence competitive performance. Health metrics including injury history, illness status, and recovery indicators are similarly unavailable but critically important for understanding an athlete's true competitive potential.

Psychological factors represent another major data gap. Confidence levels, pressure-handling ability, and competitive experience vary dramatically between athletes but are rarely quantified in ways that can be incorporated into predictive models. Environmental data such as altitude effects, weather conditions, and travel fatigue add another layer of missing information that could significantly impact model accuracy.

3. #### Validation and Generalization Issues
Traditional cross-validation approaches may not adequately reflect real-world temporal dependencies that exist in swimming performance data. Meet-level clustering effects, where performances at the same meet tend to be more similar due to shared conditions, are not properly accounted for in current validation approaches. Swimmer-level dependencies, where multiple performances from the same athlete are not truly independent observations, create potential data leakage risks that may inflate apparent model performance.

External validity concerns raise questions about model generalization across different contexts. Models trained on specific types of meets may not generalize effectively to other competition formats. Geographic and cultural biases present in training data may limit model applicability across different competitive environments. Equipment changes and rule modification, such as suppersuits and techniques, that occur over time can affect model relevance and require periodic retraining to maintain accuracy.

## Conclusion
While swimming prediction models demonstrate strong statistical performance in controlled conditions, users must understand their fundamental limitations. The inherent unpredictability of human athletic performance, combined with data availability constraints and methodological limitations, means these models should be viewed as sophisticated tools rather than definitive predictions.
