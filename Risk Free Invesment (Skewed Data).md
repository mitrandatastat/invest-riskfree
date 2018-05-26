
### Invoke Packages


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
%matplotlib inline
```

### Reading dataframe and clearning the noise


```python
loans_2007 = pd.read_csv(r"C:\Users\Yogi_Ashwast\Desktop\Temporary Parking\Data Science\Project_Work\Credit Risk Modeling\databank\loans_2007.csv")

# Removing rows with duplicate data
loans_2007 = loans_2007.drop_duplicates()
# Display first row of  the dataframe
print(loans_2007.info())
# Display dataframe size
print("\nTotal number of features : ",  loans_2007.shape[1])
```

    C:\Users\Yogi_Ashwast\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 42538 entries, 0 to 42537
    Data columns (total 52 columns):
    id                            42538 non-null object
    member_id                     42535 non-null float64
    loan_amnt                     42535 non-null float64
    funded_amnt                   42535 non-null float64
    funded_amnt_inv               42535 non-null float64
    term                          42535 non-null object
    int_rate                      42535 non-null object
    installment                   42535 non-null float64
    grade                         42535 non-null object
    sub_grade                     42535 non-null object
    emp_title                     39911 non-null object
    emp_length                    42535 non-null object
    home_ownership                42535 non-null object
    annual_inc                    42531 non-null float64
    verification_status           42535 non-null object
    issue_d                       42535 non-null object
    loan_status                   42535 non-null object
    pymnt_plan                    42535 non-null object
    purpose                       42535 non-null object
    title                         42523 non-null object
    zip_code                      42535 non-null object
    addr_state                    42535 non-null object
    dti                           42535 non-null float64
    delinq_2yrs                   42506 non-null float64
    earliest_cr_line              42506 non-null object
    inq_last_6mths                42506 non-null float64
    open_acc                      42506 non-null float64
    pub_rec                       42506 non-null float64
    revol_bal                     42535 non-null float64
    revol_util                    42445 non-null object
    total_acc                     42506 non-null float64
    initial_list_status           42535 non-null object
    out_prncp                     42535 non-null float64
    out_prncp_inv                 42535 non-null float64
    total_pymnt                   42535 non-null float64
    total_pymnt_inv               42535 non-null float64
    total_rec_prncp               42535 non-null float64
    total_rec_int                 42535 non-null float64
    total_rec_late_fee            42535 non-null float64
    recoveries                    42535 non-null float64
    collection_recovery_fee       42535 non-null float64
    last_pymnt_d                  42452 non-null object
    last_pymnt_amnt               42535 non-null float64
    last_credit_pull_d            42531 non-null object
    collections_12_mths_ex_med    42390 non-null float64
    policy_code                   42535 non-null float64
    application_type              42535 non-null object
    acc_now_delinq                42506 non-null float64
    chargeoff_within_12_mths      42390 non-null float64
    delinq_amnt                   42506 non-null float64
    pub_rec_bankruptcies          41170 non-null float64
    tax_liens                     42430 non-null float64
    dtypes: float64(30), object(22)
    memory usage: 17.2+ MB
    None
    
    Total number of features :  52
    

### Cleaning dataframe by columns carrying unuseful information


```python
cols = ["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d", "zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"]
loans_2007 = loans_2007.drop(cols, axis=1)
loans_2007.shape[1]
```




    32



### Selecting Target Column  


```python
# Selecting "loan_status" as a target column
loans_2007["loan_status"].value_counts()
```




    Fully Paid                                             33136
    Charged Off                                             5634
    Does not meet the credit policy. Status:Fully Paid      1988
    Current                                                  961
    Does not meet the credit policy. Status:Charged Off      761
    Late (31-120 days)                                        24
    In Grace Period                                           20
    Late (16-30 days)                                          8
    Default                                                    3
    Name: loan_status, dtype: int64



### Cleaning the target column


```python
# Only first two values above in "Load_Status" are found to be useful for our modeling.
keep = ["Fully Paid", "Charged Off"]
map_dict = {"Fully Paid": 1, "Charged Off": 0}

loans_2007 = loans_2007[(loans_2007["loan_status"] == keep[0]) | (loans_2007["loan_status"] == keep[1])]
loans_2007["loan_status"] = loans_2007["loan_status"].map(map_dict)
loans_2007["loan_status"].unique()
```




    array([1, 0], dtype=int64)



### Removing the columns with single value


```python
# Since single valued columns carry no useful information for credibility evaluation purpose.
drop_columns = []

for col in loans_2007.columns:
    current = loans_2007[col].dropna()
    if len(current.unique()) == 1:
        drop_columns.append(col)

loans_2007 = loans_2007.drop(drop_columns, axis=1)
loans_2007.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
           'home_ownership', 'annual_inc', 'verification_status', 'loan_status',
           'purpose', 'title', 'addr_state', 'dti', 'delinq_2yrs',
           'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'last_credit_pull_d',
           'pub_rec_bankruptcies'],
          dtype='object')



### Preserving the partially cleaned dataframe in .csv format


```python
loans_2007.to_csv("filtered_loans_2007.csv", index=False)
```

### Checking the dataframe for null values


```python
# Display list of columns carrying null values (in descending order)
null_counts = loans_2007.isnull().sum().sort_values(ascending=False)
null_counts[0:10]
```




    pub_rec_bankruptcies    697
    revol_util               50
    title                    10
    last_credit_pull_d        2
    purpose                   0
    term                      0
    int_rate                  0
    installment               0
    emp_length                0
    home_ownership            0
    dtype: int64



### Verifying the null values proportion 


```python
# Show columns with number of null values as percentage of total training examples (in descending order) 
null_percent = (loans_2007.isnull().sum()/loans_2007.shape[0]).sort_values(ascending=False)*100
null_percent[0:5]
```




    pub_rec_bankruptcies    1.797782
    revol_util              0.128966
    title                   0.025793
    last_credit_pull_d      0.005159
    purpose                 0.000000
    dtype: float64



Consider clearning columns with more less 1% NULL value while removing the columns with above 1% NULL values.

### Further Cleaning/Removing the dataframe


```python
loans_2007 = loans_2007.drop('pub_rec_bankruptcies', axis=1)
loans_2007 = loans_2007.dropna(subset=['title', 'revol_util', 'last_credit_pull_d'], axis=0, how='any')
loans_2007.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
           'home_ownership', 'annual_inc', 'verification_status', 'loan_status',
           'purpose', 'title', 'addr_state', 'dti', 'delinq_2yrs',
           'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'last_credit_pull_d'],
          dtype='object')



### Review the columns that carry object type data


```python
object_columns_df = loans_2007.select_dtypes(include=["object"])
object_columns_df.iloc[0]
```




    term                     36 months
    int_rate                    10.65%
    emp_length               10+ years
    home_ownership                RENT
    verification_status       Verified
    purpose                credit_card
    title                     Computer
    addr_state                      AZ
    earliest_cr_line          Jan-1985
    revol_util                   83.7%
    last_credit_pull_d        Jun-2016
    Name: 0, dtype: object



The following columns seem to carry categorical data that could be helpful for our regressional analysis later:

* term
* emp_length
* home_ownership
* verification_status
* purpose
* title

The above results shows that the following columns carry type of information that we can convert into numerical type for further analysis:

* int_rate
* revol_util

The remaining table seem to be not much useful and should be drop for our dataframe. They are: 

* addr_state
* earliest_cr_line
* last_credit_pull_d

### Verify the columns with categorical data 


```python
cols = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'title']

for i in cols:
    print("\nUnique Counts for " + i + " :\n", object_columns_df[i].value_counts())
```

    
    Unique Counts for term :
      36 months    29041
     60 months     9667
    Name: term, dtype: int64
    
    Unique Counts for emp_length :
     10+ years    8545
    < 1 year     4513
    2 years      4303
    3 years      4022
    4 years      3353
    5 years      3202
    1 year       3176
    6 years      2177
    7 years      1714
    8 years      1442
    9 years      1229
    n/a          1032
    Name: emp_length, dtype: int64
    
    Unique Counts for home_ownership :
     RENT        18513
    MORTGAGE    17112
    OWN          2984
    OTHER          96
    NONE            3
    Name: home_ownership, dtype: int64
    
    Unique Counts for verification_status :
     Not Verified       16696
    Verified           12290
    Source Verified     9722
    Name: verification_status, dtype: int64
    
    Unique Counts for purpose :
     debt_consolidation    18130
    credit_card            5039
    other                  3864
    home_improvement       2897
    major_purchase         2155
    small_business         1762
    car                    1510
    wedding                 929
    medical                 680
    moving                  576
    vacation                375
    house                   369
    educational             320
    renewable_energy        102
    Name: purpose, dtype: int64
    
    Unique Counts for title :
     Debt Consolidation                                            2104
    Debt Consolidation Loan                                       1632
    Personal Loan                                                  642
    Consolidation                                                  494
    debt consolidation                                             485
    Credit Card Consolidation                                      353
    Home Improvement                                               346
    Debt consolidation                                             324
    Small Business Loan                                            310
    Credit Card Loan                                               305
    Personal                                                       302
    Consolidation Loan                                             251
    Home Improvement Loan                                          234
    personal loan                                                  227
    personal                                                       211
    Loan                                                           208
    Wedding Loan                                                   201
    Car Loan                                                       195
    consolidation                                                  193
    Other Loan                                                     181
    Credit Card Payoff                                             150
    Wedding                                                        149
    Credit Card Refinance                                          143
    Major Purchase Loan                                            139
    Consolidate                                                    125
    Medical                                                        118
    Credit Card                                                    115
    home improvement                                               107
    My Loan                                                         92
    Credit Cards                                                    91
                                                                  ... 
    re-finance                                                       1
    ktm                                                              1
    2010 taxes                                                       1
    To finance tax payments                                          1
    tiffinoutofdebtnow                                               1
    remodel bathroom                                                 1
    Athomas74                                                        1
    Refinance Exec.                                                  1
    One Monthly Payment Instead of Many                              1
    Help Make My Dream A Reality                                     1
    Dennell                                                          1
    Paul Family Loan                                                 1
    School for the Whole Family                                      1
    Getting Married Thanksgiving Weekend!                            1
    Todd                                                             1
    Without Prejudice                                                1
    Credit cards are ripping me off! Help me stick it to them!       1
    AL LOAN                                                          1
    Debt-Free                                                        1
    Baby Makes Three and No Room .......                             1
    keywest                                                          1
    Organizing many payments after divorce                           1
    KMS Loan                                                         1
    Tuff Luv                                                         1
    MRC Move                                                         1
    Need to Move and Debt Consolidation                              1
    Discover payoff                                                  1
    Paypal Credit Loan                                               1
    Want to Get a Pellet Stove                                       1
    Consolidating bills to go to school                              1
    Name: title, Length: 19332, dtype: int64
    

It appears that column 'purpose' and 'title' carry overlapping information. However, considering the duplication of loan categories under the 'title' column, let's keep the 'purpose' column and drop the 'title' one.

Column 'emp_length' carries multiple category data.

### Drop less useful columns with duplicate/overlapping data


```python
cols = ['addr_state', 'earliest_cr_line', 'last_credit_pull_d', 'title']
loans_2007 = loans_2007.drop(cols, axis=1)
```

### Assign data variables to columns


```python
cat_columns = ['home_ownership', 'verification_status', 'purpose', 'term']
loans_2007_dummies = pd.get_dummies(loans_2007[cat_columns])
loans_2007 = pd.concat([loans_2007, loans_2007_dummies], axis=1)

# Removinng the original non-dummy columns 
loans_2007 = loans_2007.drop(cat_columns, axis=1)
loans_2007.columns
```




    Index(['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc',
           'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
           'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'home_ownership_MORTGAGE', 'home_ownership_NONE',
           'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
           'verification_status_Not Verified',
           'verification_status_Source Verified', 'verification_status_Verified',
           'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
           'purpose_educational', 'purpose_home_improvement', 'purpose_house',
           'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
           'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding', 'term_ 36 months',
           'term_ 60 months'],
          dtype='object')



### Mapping column with multiple category data


```python
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans_2007 = loans_2007.replace(mapping_dict)
np.sort(loans_2007['emp_length'].unique().astype('int64'))
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=int64)



### Convert Columns String type Numeric Data to Real Numbers


```python
loans_2007['int_rate'] = loans_2007['int_rate'].str.rstrip('%').astype('float')
loans_2007['revol_util'] = loans_2007['revol_util'].str.rstrip('%').astype('float')
```

### Preserving completely cleansed dataframe in .csv format


```python
loans_2007.to_csv("cleaned_loans_2007.csv", index=False)
loans_2007.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 38708 entries, 0 to 39785
    Data columns (total 38 columns):
    loan_amnt                              38708 non-null float64
    int_rate                               38708 non-null float64
    installment                            38708 non-null float64
    emp_length                             38708 non-null int64
    annual_inc                             38708 non-null float64
    loan_status                            38708 non-null int64
    dti                                    38708 non-null float64
    delinq_2yrs                            38708 non-null float64
    inq_last_6mths                         38708 non-null float64
    open_acc                               38708 non-null float64
    pub_rec                                38708 non-null float64
    revol_bal                              38708 non-null float64
    revol_util                             38708 non-null float64
    total_acc                              38708 non-null float64
    home_ownership_MORTGAGE                38708 non-null uint8
    home_ownership_NONE                    38708 non-null uint8
    home_ownership_OTHER                   38708 non-null uint8
    home_ownership_OWN                     38708 non-null uint8
    home_ownership_RENT                    38708 non-null uint8
    verification_status_Not Verified       38708 non-null uint8
    verification_status_Source Verified    38708 non-null uint8
    verification_status_Verified           38708 non-null uint8
    purpose_car                            38708 non-null uint8
    purpose_credit_card                    38708 non-null uint8
    purpose_debt_consolidation             38708 non-null uint8
    purpose_educational                    38708 non-null uint8
    purpose_home_improvement               38708 non-null uint8
    purpose_house                          38708 non-null uint8
    purpose_major_purchase                 38708 non-null uint8
    purpose_medical                        38708 non-null uint8
    purpose_moving                         38708 non-null uint8
    purpose_other                          38708 non-null uint8
    purpose_renewable_energy               38708 non-null uint8
    purpose_small_business                 38708 non-null uint8
    purpose_vacation                       38708 non-null uint8
    purpose_wedding                        38708 non-null uint8
    term_ 36 months                        38708 non-null uint8
    term_ 60 months                        38708 non-null uint8
    dtypes: float64(12), int64(2), uint8(24)
    memory usage: 5.3 MB
    

# Feature Scaling


```python
col_scale = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc']

for i in range(len(col_scale)):
    loans_2007[col_scale[i]] = (loans_2007[col_scale[i]] - np.mean(loans_2007[col_scale[i]])) / (np.std(loans_2007[col_scale[i]]))

print(loans_2007[col_scale].head(3))
```

       loan_amnt  int_rate  installment  emp_length  annual_inc       dti  \
    0  -0.825582 -0.349656    -0.766989    1.440396   -0.698714  2.152857   
    1  -1.165155  0.900136    -1.260922   -1.337582   -0.605205 -1.840154   
    2  -1.178738  1.086793    -1.143479    1.440396   -0.881806 -0.683455   
    
       delinq_2yrs  inq_last_6mths  open_acc  
    0    -0.298043        0.120060 -1.428405  
    1    -0.298043        3.852801 -1.428405  
    2    -0.298043        1.053245 -1.655675  
    

### Evaluating the feature correlations with the target 


```python
corr_mat = loans_2007.corr()['loan_status'].abs().sort_values(ascending=False)
corr_mat
```




    loan_status                            1.000000
    int_rate                               0.208826
    term_ 60 months                        0.169424
    term_ 36 months                        0.169424
    revol_util                             0.100114
    purpose_small_business                 0.076865
    inq_last_6mths                         0.071429
    loan_amnt                              0.058938
    pub_rec                                0.051424
    dti                                    0.044914
    verification_status_Not Verified       0.042797
    verification_status_Verified           0.041952
    purpose_credit_card                    0.041195
    annual_inc                             0.040533
    purpose_major_purchase                 0.029308
    installment                            0.027611
    home_ownership_MORTGAGE                0.023232
    purpose_car                            0.022749
    total_acc                              0.022230
    purpose_home_improvement               0.021255
    home_ownership_RENT                    0.021220
    delinq_2yrs                            0.020522
    purpose_debt_consolidation             0.020002
    purpose_wedding                        0.018579
    purpose_other                          0.016760
    open_acc                               0.008596
    emp_length                             0.007786
    purpose_educational                    0.007762
    revol_bal                              0.006347
    purpose_renewable_energy               0.006016
    home_ownership_OTHER                   0.006009
    purpose_moving                         0.005117
    purpose_house                          0.004132
    verification_status_Source Verified    0.003844
    home_ownership_NONE                    0.003626
    purpose_medical                        0.003551
    home_ownership_OWN                     0.002514
    purpose_vacation                       0.001047
    Name: loan_status, dtype: float64



### Develop Logistic Regression Model


```python
features = loans_2007.columns.values
# split train/test/cv as 60/20/20
train_set = loans_2007.iloc[:23224]
test_set = loans_2007.iloc[23225:30966]
cross_val = loans_2007.iloc[30967:]
predictions = list()
mse_vals = []
k = []

# Finding the optimal features for the model
for i in range(len(features)):
    logistic_model = LogisticRegression()
    logistic_model.fit(train_set[features[0:i]], train_set["loan_status"])
    predictions.append(logistic_model.predict(test_set[features[0:i]]))
    mse_val.append(mean_squared_error(predictions[i], test_set["loan_status"]))
    k.append = i

optimal_features = k[mse_val.index(min(mse_val))]
```

    loan_amnt
        loan_amnt  int_rate  installment  emp_length  annual_inc  loan_status  \
    0   -0.825582 -0.349656    -0.766989    1.440396   -0.698714            1   
    1   -1.165155  0.900136    -1.260922   -1.337582   -0.605205            0   
    2   -1.178738  1.086793    -1.143479    1.440396   -0.881806            1   
    3   -0.146436  0.418614     0.078795    1.440396   -0.305974            1   
    5   -0.825582 -1.093580    -0.797716   -0.504188   -0.511695            1   
    6   -0.553923  1.086793    -0.732427    0.884800   -0.340199            1   
    7   -1.097240  1.811780    -1.023159    1.162598   -0.324676            1   
    8   -0.744084  2.525947    -0.817226   -0.226391   -0.449356            0   
    9   -0.774646  0.202200    -0.965540   -1.337582   -0.838978            0   
    10  -0.621838  0.732415    -0.812145    0.051407    0.049362            1   
    
             dti  delinq_2yrs  inq_last_6mths  open_acc       ...         \
    0   2.152857    -0.298043        0.120060 -1.428405       ...          
    1  -1.840154    -0.298043        3.852801 -1.428405       ...          
    2  -0.683455    -0.298043        1.053245 -1.655675       ...          
    3   1.006645    -0.298043        0.120060  0.162485       ...          
    5  -0.311872    -0.298043        1.986430 -0.064785       ...          
    6   1.532554    -0.298043        0.120060 -0.519325       ...          
    7  -1.188387    -0.298043        1.053245 -1.201135       ...          
    8  -1.158421    -0.298043        1.053245  0.389755       ...          
    9   0.718969    -0.298043       -0.813126 -1.655675       ...          
    10  0.425299    -0.298043        1.053245  1.071565       ...          
    
        purpose_major_purchase  purpose_medical  purpose_moving  purpose_other  \
    0                        0                0               0              0   
    1                        0                0               0              0   
    2                        0                0               0              0   
    3                        0                0               0              1   
    5                        0                0               0              0   
    6                        0                0               0              0   
    7                        0                0               0              0   
    8                        0                0               0              0   
    9                        0                0               0              1   
    10                       0                0               0              0   
    
        purpose_renewable_energy  purpose_small_business  purpose_vacation  \
    0                          0                       0                 0   
    1                          0                       0                 0   
    2                          0                       1                 0   
    3                          0                       0                 0   
    5                          0                       0                 0   
    6                          0                       0                 0   
    7                          0                       0                 0   
    8                          0                       1                 0   
    9                          0                       0                 0   
    10                         0                       0                 0   
    
        purpose_wedding  term_ 36 months  term_ 60 months  
    0                 0                1                0  
    1                 0                0                1  
    2                 0                1                0  
    3                 0                1                0  
    5                 1                1                0  
    6                 0                0                1  
    7                 0                1                0  
    8                 0                0                1  
    9                 0                0                1  
    10                0                0                1  
    
    [10 rows x 38 columns]
    




    '\n\n# Finding the optimal features for the model\nfor i in range(len(features)):\n    logistic_model = LogisticRegression()\n    logistic_model.fit(train_set[features[0:i]], train_set["loan_status"])\n    predictions.append(logistic_model.predict(test_set[features[0:i]]))\n    mse_val.append(mean_squared_error(predictions[i], test_set["loan_status"]))\n    k.append = i\n\noptimal_features = k[mse_val.index(min(mse_val))]\n'


