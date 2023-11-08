# Payment Transaction Volumes Forecasting

#### -- Project Status: [Completed]

## Project Description and Business Case
In a business driven by the number of transactions, historical volumes provide a solid foundation for setting future revenue expectations, yet overall trends may hide valuable insights through combining opposing signals. Therefore, an informed analysis of merchant level fluctuations is needed to design a reliable forecasting mechanism that could serve the business purpose.

This project was given to us by a Swiss payment processing company as our final project for our 480 hours Data Science Boootcamp by SIT Learning. The company connects financial markets in Switzerland and abroad, providing services related to its four business units: stock exchanges, securities services, banking services, and financial information data provider. The company’s payment processing revenue model is based on transaction fees earned each time a merchant receives a payment and varies depending on the service that the company is performing (i.e. direct transaction fees for standard payments, increased 3-D secure checkout, and fraud check).

## Project Goals
Our goal was to build a fully automated pipeline to forecast future transactions while analysing the business context behind the provided dataset.

## Milestones
* Explore the historical data, derive conclusions on the trends and visualise findings
* Propose forecasting approaches together with validation mechanisms
* Develop forecasting models, assess their performance and fine tune
* Develop a holistic solution for ongoing forecasting and validation of the time series
* Presentation of the results
* Validate designed solution on new data points

### Methods Used
* Exploratory data analysis
* Time series modelling: SARIMA, ETS, and Prophet
* Explored clustering methods: K-nn for time series, t-SNE, functional PCA, PyCaret, SOM, and TSlearn
* Visualization with Plotly

### Technologies
* Python
* Pandas, Scikit-lean, Numpy, Datetime, Scipy, Plotly, Matplotlib, Seaborn
* Pmdarima, Prophet, Statsmodels, Multiprocess

## How to run
Given monthly data of transactions, the following scripts can be used to create models and predictions:
* src/run_all_three.py
=> Creates and finetunes 3 models - based on ARIMA, ETS and PROPHET (this is good when you do not have many data, and would like to track these models for longer time to see how they perform)

* src/run_best_model.py
=> Based on train-test split you can change inside the code, the best of the 3 fine tuned model is selected, and then used to create the forecast

After the predictions have been created, you can run the following script:
* src/update_errors.py
=> Takes the mock data with additional month and calculates the RMSE for the predictions that have been saved vs the new data. These values are then saved into file data/all_errors.csv

If you just want to track the errors of the models you have created, you can run the following script:
* update_errors.py
=> Run this script every time you get new monthly data

You can also rerun run_all_three.py or run_best_model.py every month you get new data and create new models. When again the following monthly data come in, you will just need to run update_errors.py. These new errors will be added to the all_errors.csv.

There is also possibility to create a subgroup of merchants based on their total number of transactions, either the top or bottom merchants that make up X% of all the transactions. This can be done by running the following script :
* src/create_sub_group_totals.py
=> Sample outputs are also saved in src/data/raw/

The run_all_three.py or the run_best_model.py can be also used on individual merchants or totals of subgroup of merchants.

## Streamlit App
Our team created a Streamlit App that runs the scrips for visual representation:
* [Streamlit App](https://transaction-volumes-forecasting-app-py.streamlit.app/)
* [Mock data](https://github.com/mikjf/transaction_volumes_forecasting_streamlit_app/blob/main/mock_data/Mock_Time_Series_Merchants_Transactions_Anonymized.csv)
* [Streamlit App GitHub](https://github.com/mikjf/transaction_volumes_forecasting_streamlit_app)

## Journal
* [Journal details](JOURNAL.txt)

## Featured Notebooks
* [Modelling Notebooks](notebooks/)

## Contributing Members
* [Alžbeta Bohiniková](https://github.com/Betka112)
* [Luis Miguel Rodríguez Sedano](https://github.com/Euphorbix)
* [Mukund Pondkule](https://github.com/mpondkule)
* [Michael Flury](https://github.com/mikjf)

Project Organization
------------

    ├── JOURNAL.txt                                     # Journal details
    ├── LICENSE
    ├── Makefile
    ├── README.md                                       # Project details
    ├── data
    │   └── error_data
    ├── docs
    │   ├── Makefile
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   └── make.bat
    ├── notebooks                                       # Team notebooks (final versions and members' versions)
    │   ├── Betka
    │   ├── Final notebooks
    │   ├── Luis
    │   ├── Michael
    │   └── Mukund
    ├── references
    ├── reports                                         # Project report presentation
    ├── requirements.txt                                # Requirements to install to run the scripts
    ├── setup.py
    ├── src                                             # Source for scripts
    │   ├── ETS_Function.py
    │   ├── Visualization.py
    │   ├── auto_arima_single.py
    │   ├── auto_single_prophet.py
    │   ├── create_sub_group_totals.py
    │   ├── data
    │   │   ├── all_errors.csv
    │   │   ├── processed                               # Processed data (output) appear in this folder
    │   │   └── raw                                     # Mock data to run the scripts
    │   ├── experimental_multivariate_prophet_ets       # Experiment using individual merchants as regressors   
    │   ├── features
    │   ├── get_sum.py
    │   ├── load_and_output_top_merchants.py
    │   ├── models                                      # Pkl/JSON format models
    │   ├── run_all_three.py
    │   ├── run_arima_tune_get_pred.py
    │   ├── run_best_model.py
    │   ├── run_prophet_tune_get_pred.py
    │   ├── tracking_files                              # JSON tracking files
    │   ├── update_errors.py
    └── tox.ini


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
