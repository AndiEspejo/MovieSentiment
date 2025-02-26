# MovieSentiment: Natural Language Processing for Film Reviews

## Project Overview

MovieSentiment is an NLP-based sentiment analysis system designed to automatically classify movie reviews as positive or negative. This project demonstrates the power of natural language processing techniques to understand human sentiment in written text, specifically applied to the entertainment domain.

Using machine learning algorithms trained on a dataset of labeled movie reviews, this system can determine whether a review expresses a favorable or unfavorable opinion about a film with high accuracy.

## Key Features

- **Binary Sentiment Classification**: Accurately categorizes reviews as positive or negative
- **Comprehensive Text Preprocessing**: Implements tokenization, stopword removal, stemming, and lemmatization
- **Feature Engineering**: Transforms text data into meaningful numerical features
- **Multiple ML Models**: Implements and compares various classification algorithms
- **Performance Metrics**: Detailed evaluation using accuracy, F1-score, precision, and recall
- **Visualization**: Intuitive graphs to understand model performance and data distribution

## Technology Stack

- **Python**: Core programming language
- **NLTK**: Natural Language Toolkit for text preprocessing
- **Scikit-learn**: Machine learning models implementation
- **Pandas/NumPy**: Data manipulation and numerical processing
- **Matplotlib/Seaborn**: Data visualization
- **Spacy**: Advanced NLP capabilities including lemmatization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-sentiment.git
cd movie-sentiment

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Download Spacy model
python -m spacy download en_core_web_sm
```

## Project Structure

```
movie-sentiment/
├── data/
│   ├── movies_review_train.csv
│   └── movies_review_test.csv
├── notebooks/
│   └── MovieSentiment.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── requirements.txt
└── README.md
```

## Data

The project uses a curated dataset of movie reviews labeled with sentiment (positive/negative). The data is split into:

- **Training set**: Used to train the classification models
- **Testing set**: Used to evaluate model performance on unseen data

Features include the review text and the sentiment label (binary: positive/negative).

## Preprocessing Pipeline

The text preprocessing pipeline includes:

1. **Normalization**: Converting to lowercase, removing special characters
2. **Tokenization**: Using NLTK's ToktokTokenizer
3. **Stopword Removal**: Filtering common English words using NLTK's stopwords list
4. **Stemming**: Reducing words to their root form using Porter Stemmer
5. **Lemmatization**: Converting words to their base form using Spacy's en_core_web_sm model

## Model Development

The project explores various classification approaches:

- Naive Bayes classifiers
- Logistic Regression
- Support Vector Machines
- Random Forest
- Gradient Boosting

Text is vectorized using:

- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word embeddings

## Performance

The best-performing model achieves:

- **Accuracy**: 89.7%
- **F1 Score**: 0.896
- **Precision**: 0.883
- **Recall**: 0.910

## Usage

To run the project notebook:

```bash
jupyter notebook notebooks/MovieSentiment.ipynb
```

To run the tests:

```bash
pytest tests/
```

## Future Enhancements

- Implement deep learning models (LSTM, BERT)
- Add multiclass sentiment analysis (beyond binary classification)
- Develop a web interface for real-time review analysis
- Explore aspect-based sentiment analysis for specific movie features (acting, plot, visuals)
- Create a recommendation system based on sentiment patterns

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the project.

## License

MIT
