# Reference
Implementing algorithms from scratch so I can learn more about machine learning, statistics and computer science.

## Notebooks

### Machine Learning
- Machine Learning Recipes [[1](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)]
    - [Decision Trees and visualising decisions](notebooks/ml_recipes_1.ipynb)
    - [Implementing KNN and Euclidean distance from scratch](notebooks/ml_recipes_2.ipynb)
- Programmers Guide to Data Mining [[1](http://guidetodatamining.com/)]
    - [Collaborative filtering: pearson correlation, cosine similarity, KNN](notebooks/programmers_guide_1.ipynb)
    - [Implicit ratings and item based filtering](notebooks/programmers_guide_2.ipynb)
    - [Classification based on item attributes](notebooks/programmers_guide_3.ipynb)
    - [Evaluating Algorithms](notebooks/programmers_guide_4.ipynb)
    - [Naïve Bayes](notebooks/programmers_guide_5.ipynb)
    - [Unstructured Text](notebooks/programmers_guide_6.ipynb)
    - [Clustering](notebooks/programmers_guide_7.ipynb)
- Practical Machine Learning [[1](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)]
    - [Sklearn regression, SVM & pickling with share market data](notebooks/practical_ml_1.ipynb)
    - [Implementing regression from scratch](notebooks/practical_ml_2.ipynb)
- [Unsupervised Machine Learning: k-means & mean-shift](notebooks/unsupervised_ml.ipynb) [[1](https://pythonprogramming.net/flat-clustering-machine-learning-python-scikit-learn/)]

### Deep Learning
- [Intro to Keras](notebooks/keras_intro.ipynb) [[1](http://machinelearningmastery.com/introduction-python-deep-learning-library-keras/), [2](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)]
- [Installing Keras with Tensorflow and CUDA](keras_install_cuda_tf.md)

### Natural Language Processing
- [Machine Learning with Text: Sklearn & bag of words](notebooks/ml_text.ipynb) [[1](https://www.youtube.com/watch?v=vTaxdJ6VYWE)]
- [NLTK with Python 3 for Natural Language Processing](notebooks/natural_language.ipynb) [[1](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL), [2](https://www.youtube.com/watch?v=itKNpCPHq3I)]

### Statistics
- [Bayes Made Simple](notebooks/bayes_simple.ipynb) [[1](https://www.youtube.com/watch?v=6GV5bTCLC8g), [2](http://greenteapress.com/wp/think-bayes/), [3](https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/), [4](https://www.springboard.com/blog/probability-bayes-theorem-data-science/)]
- [Monte Carlo Simulation with Python](notebooks/monte_carlo_intro.ipynb) [[1](https://www.youtube.com/playlist?list=PLQVvvaa0QuDdhOnp-FnVStDsALpYk2hk0)]

### Programming
- [Web Scraping with Beautiful Soup](notebooks/beautiful_soup.ipynb) [[1](https://www.dataquest.io/blog/web-scraping-tutorial-python/)]
- [Python Multiprocessing](notebooks/py_multiprocess.ipynb) [[1](https://youtu.be/oEYDqQ1pq9o), [2](https://youtu.be/kUKOEuPJXGc)]
- [Python Decorators](notebooks/py_decorators.ipynb) [[1](https://www.youtube.com/watch?v=rPCeCPT-f28&list=LLuei0qkBoeOass8xV_cOrqQ&index=1)]
- [MyPy Python Type Checker](notebooks/my_py.ipynb) [[1](http://mypy-lang.org/)]
- [Python Base64 Encode/Decode & PIL](notebooks/py_base64.ipynb)
- [Python Args & Kwargs](notebooks/args_kwargs.ipynb) [[1](https://youtu.be/gZB_ENJD34E)]
- [Python Logging](notebooks/python_logging.ipynb) [[1](https://youtu.be/-RcDmGNSuvU)]
- [SQL in the Jupyter Notebook](notebooks/ipython_sql.ipynb)

## System Notes

- Python [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/): `virtualenv venv -> cd $DIR -> source venv/bin/activate`
- Updating all Python packages `pip3 freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip3 install -U`
- Homebrew update and upgrade: `brew update && brew upgrade`
- Reload module and output version information
```
%load_ext autoreload
%autoreload 2

%load_ext version_information
%version_information numpy, scipy, matplotlib, pandas
```
- Run bash in the Jupyter notebook
```
%%bash
curl http://google.com
```