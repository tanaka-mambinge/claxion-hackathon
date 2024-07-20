# Hackathon notes

## Training

- I tested the logistic regressor, decision, random forest and xgboost models. The random forest and xgboost gave the best performances after running grid search to find the best possible hyperparameters for the models. Cross validation was handled by the grid search class provided by scikit-learn.

## Model selection

- I chose the Random forest as the best model mainly because after multiple iterations, it returned a lower false positive and false negative rate. The lower the false positive rate, the lower the likelihood of misclassifying a person who likely to default on a loan and this should save the business money form loaning out money to people who are likely to default on their loans

## Model drift detection

- Found it hard to get a good threshold to detect if the new data is significantly different from the past data

## API design choices

- A simpler model training endpoint had to be developed for hackathon purposes. But in practice, data would be periodically collected and stored. A pipeline and orchestration library would be used to schedule model training jobs and also check if there's model drift. Scheduling jobs would be a better approach over using the API because a timeout occur and stop the training job. Scheduling a job has the advantage that training can happen in the background and the user can poll the server, use a webhook or get email notifications once the training job is done.
- For model inference, using JSON would be ideal if the API is to be connected to a website. A website would be more ideal as data entry errors would be reduce as users can select categorical data from a list e.g select the person's job instead of typing it manually in a program like excel.

## API usage

### Model inference

- The inference endpoint (http://127.0.0.1:8000/api/v1/model/inference) will assume the data has already been cleaned and transformations have already been applied to it
- Upload a csv file with the following columns: `is_employed, loan_amount, number_of_defaults, outstanding_balance, interest_rate, age, salary, gender, location, job, marital_status`. Check [column details](#column-details) for more details.
- The endpoint will respond with your csv file with an additional `status` column.

### Model training

- The model training endpoint (http://127.0.0.1:8000/api/v1/model/train) will assume the data has already been cleaned and only a few preprocessing steps have to be made e.g. one-hot encoding categorical features
- Upload a csv file with the following columns: `is_employed, loan_amount, number_of_defaults, outstanding_balance, interest_rate, age, salary, gender, location, job, marital_status, defaulted`. Check [column details](#column-details) for more details.
- The endpoint will return a model report and the trained model will be saved in the artifacts folder using the following filename format `{algo name}-{macro f1 score}-{weighted f1 score}{timestamp}.sav`

### Column Details

- `is_employed` - A boolean value indicating whether a person is employed or not. Accepted values are:
  - `true` for employed
  - `false` for unemployed
- `loan_amount` - A number representing the amount of money the person loaned e.g. 100.00
- `number_of_defaults` - A number representing the number of times a person defaulted on a loan e.g. 1
- `outstanding_balance` - A number representing the outstanding balance
- `interest_rate` - A nubmer representing the interest rate of the loan
- `age` - A number representing the age of the person taking the loan
- `salary` - A number representing the salary of the person taking the loan
- `gender` - A string representing the gender of the person taking the loan. Using a value not in the list will replace the value with other. Accepted values are: `male`, `female`, `other`
- `location` - A string representing the location of the person taking the loan. Using a value not in the list will replace the value with other. Accepted values are: `kariba`, `beitbridge`, `harare`, `kadoma`, `chimanimani`, `gweru`, `hwange`, `shurugwi`, `karoi`, `chiredzi`, `kwekwe`, `rusape`, `marondera`, `zvishavane`, `mutare`, `plumtree`, `victoria_falls`, `nyanga`, `redcliff`, `chipinge`, `bulawayo`, `masvingo`, `chivhu`, `gokwe`, `other`
- `job` - A boolean value indicating whether a person is employed or not. Using a value not in the list will replace the value with other. Accepted values are: `data_analyst`, `software_developer`, `nurse`, `engineer`, `teacher`, `lawyer`, `doctor`, `accountant`, `data_scientist`, `other`
- `marital_status` - A string representing the marital status of the person taking the loan. Using a value not in the list will replace the value with other. Accepted values: `single`, `divorced`, `married`, `other`
- `defaulted` - A boolean value indicating whether a person is defaulted or not. Accepted values are:
  - `1` for defaulted
  - `0` for did not default
