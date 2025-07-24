# ProphitBet Dependencies

## Currently Installed Packages
- flask==3.1.1
- werkzeug==3.1.3
- pandas==2.3.1
- numpy==2.3.1
- scikit-learn==1.7.1
- joblib==1.5.1
- matplotlib==3.10.3
- requests==2.32.4
- beautifulsoup4==4.13.4
- selenium==4.34.2

## Missing Dependencies (Need Installation)
- tqdm (for progress bars in data download)
- seaborn (for enhanced data visualization)
- fuzzywuzzy (for team name matching)
- python-levenshtein (for faster string matching)

## Optional Advanced Features (Large packages)
- tensorflow (for neural networks - ~650MB)
- optuna (for hyperparameter optimization)
- imbalanced-learn (for handling imbalanced datasets)
- undetected-chromedriver (for advanced web scraping)

## Installation Commands
```bash
# Install missing core dependencies
pip install tqdm seaborn fuzzywuzzy python-levenshtein

# Install advanced features (optional)
pip install tensorflow optuna imbalanced-learn undetected-chromedriver
```