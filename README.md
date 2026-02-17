**Capstone Project**  
**Customer Churn Prediction**  
**Final Report**

**1.	Problem Statement**
   
The problem at hand is how to identify at an early stage the customers who are likely to churn so that proactive measures can be implemented to prevent these customers from leaving. We would like to predict whether a customer will churn or not churn based on certain attributes like customer satisfaction scores, how long the customer has been with the company, payment failures, customer location, customer engagement level and the contract type.
It is critical to find the reasons why customers stop using the company's products. When these factors are known, the customers who are likely to churn can be identified early enough and special programs can be implemented to prevent these customers from churning. In this way, the company's continued profitability is asssured.
Customer retention makes a lot of business sense - it costs around five (5) times more to acquire new customers than to retain existing customers. A reduction in customer churn can significantly increase revenue. The reasons for customer churn and who are these customers can be predicted before churn happens.

**2.	Model Outcomes or Predictions**
   
The type of learning is classification. The expected output of the selected model is the prediction of customers who are likely to churn and the features that drive churn. Supervised machine learning algorithms are used to build predictive models. 
The models are expected to be able to catch churners quite well so that proactive measures can be designed at an early stage to prevent them from leaving.  At the same time, a profit/loss analysis will make sure that these measures are profitable. The models will also demonstrate a capability to make useful predictions and highlight the features mostly affecting churn so that retention programs are suitably targeted.

**3.	Data**
   
The Customer Churn Prediction Business Dataset comes from Kaggle. This dataset is synthetically generated for educational, research, and portfolio purposes. While it reflects realistic business patterns, it does not represent real customer data.

**4.	Data Preprocessing/Preparation**
   
The 'customer_id' column is dropped because it does not add value to the modeling effort:  
`df.drop(columns=['customer_id'])`  
Any leading and trailing white spaces from categorical columns are removed:  
`string_cols = df.select_dtypes(include=['object']).columns`  
`df[string_cols] = df[string_cols].apply(lambda x: x.str.strip())`  
The column with missing values is identified:   
`df.columns[df.isnull().any()].tolist()`    
The percentage of missing values is calculated:   
`df['complaint_type'].isnull().sum()/df.shape[0]*100`     
The percentage of missing values is not high enough to warrant dropping the column. The missing values are instead replaced with the column mode:    
`mode_column = df['complaint_type'].mode()`  
`df['complaint_type'] = df['complaint_type'].fillna(str(mode_column))`
Inconsistent data is replaced, for example:   
`df['complaint_type'] = df['complaint_type'].replace({'0    Technical\nName: complaint_type, dtype: object':'Technical'})`  
There are no duplicate rows:  
`duplicates = len(df[df.duplicated()])`  
The distributions of categorical columns are verified:  
`categorical_cols = df.select_dtypes(include=['object']).columns.tolist()`  
`for i in categorical_cols:`
 `   value_count_column = df[i].value_counts(normalize=True)`
    print(f'The value count for column {value_count_column} \n') 
Box plots of numerical features are analyzed prior to removal of outliers to visualize the presence of outliers.
Outliers are removed from numeric features using the interquartile range (IQR) rule:
for col in numeric_list:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
Box plots of numerical features are re-analyzed after the removal of outliers. The remaining outliers are allowed to stay to conserve data. 
For the box plots, particular attention is given to when the ‘churn’ median line is lower than the ‘no churn’ median line, churn is more likely here. Overlapping ‘churn’ and ‘no churn’ boxes indicate that the feature might not be a good predictor. Separation of the churn and no churn boxes is a strong signal for good predictors.
Stacked bars of categorical features (in percentages) are plotted.    The color ratios of the stacked bars highlight features which have slightly more churn.
Histograms are plotted to visualize the distribution of numerical features to help uncover trends and patterns.
Heat maps of numeric features are displayed to highlight significant positive and negative correlations especially with ‘churn’.
A pair plot analysis of the top six (6) numeric features is made to verify the distribution of churn in these numeric features. 
The features and target variable are defined:
X = df.drop(['churn'], axis=1)
y = df['churn']
and then split into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
Scaling is not used in DecisionTreeClassifier while  LogisticRegression, KNeighborsClassifier and SVC (Support Vector Classifier) use scaling. StandardScaler is used. KNeighborsClassifier and SVC (Support Vector Classifier)are especially sensitive to the scale of input features.
Categorial features are encoded using OneHotEncoder and OrdinalEncoder.

5.	Modeling
   
The following supervised machine learning algorithms are used to build predictive models:
LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, and SVC (Support Vector Classifier)
The class imbalance is verified:
val_count_churn = df['churn'].value_counts(normalize=True)
print(val_count_churn)
churn
0    0.906262
1    0.093738
Name: proportion, dtype: float64
In order to address the class imbalance, SMOTE-NC (Synthetic Minority Oversampling Technique for Nominal and Continuous features)is used by KNeighborsClassifier while the other models use the parameter class_weight='balanced'.
Transformers like make_column_transformer and ColumnTransformer are used to prepare the data for encoding and scaling, as required, and fed to a Pipeline.
A baseline model is built to benchmark the models to be designed. Simple models of the various algorithms are initially created to further benchmark the modeling effort.
classification_report and confusion_matrix provide the Accuracy, Precision, and Recall of the models. F2 score is calculated and a Profit/Loss analysis is also made. 
The models are optimized using HalvingRandomSearchCV, GridSearchCV, and RandomizedSearchCV.
The ROC (Receiver Operating Characteristic) Curve is plotted to display the  AUC (Area Under Curve)which is a measure of the predictive power of the model and to calculate the optimal threshold using the Youden’s J method.
The Precision-Recall Curve is also plotted to demonstrate the relationship between Precision and Recall. Precision and Recall are evaluated at different thresholds.  The threshold that maximizes profit is determined.
For each classifier, models are constructed and their respective performances are compared to each other. In addition to Accuracy, Precision, Recall, F2 score, and AUC, the best model is chosen based on business goals that consider the relative cost of missing a churner (false negatives - predicted not to churn but churned, loss of lifetime value), cost of false alarms (false positives - predicted to churn but stayed, cost of retention offer) and true positives (predicted to stay and actually stayed, saved lifetime value less the cost of retention offer).
In this particular case of customer churn, missing churners (Recall) is more expensive than false alarms (precision). Recall is, therefore, optimized at the expense of precision and accuracy.
The features and their proportional importances are also identified particularly those features whose increase or decrease correspondingly raise or lower the likelihood of customer churn.
Prediction is demonstrated using samples from the test data.
Identification of customers likely to churn is also shown.

6.	Model Evaluation
   
The LogisticRegression model identified that payment failures is the number one predictor for churn. A customer with failed payments is significantly more likely to leave. A unit increase in payment_failures raises odds of churn by 47% assuming all other variables remain constant. Compared to the reference locations, customers in Germany, the UK and the city of Toronto are a high churn risk by 31%, 22%, and 19%, respectively. As these values go up, the probability of churn increases. On the other hand, csat score is the number one retention factor. High customer satisfaction is the strongest signal that a customer will stay. A unit increase in customer satisfaction scores decreases odds of churn by 42% assuming all other variables remain constant. tenure months (-40%) is another anchor - the longer a customer stays, the less likely it is to leave. monthly logins (-36%) - high product engagement is a major indicator of a healthy customer who will continue to stay. As these values go up, the probability of churn decreases.
Feature importance from the DecisionTreeClassifier model  shows that tenure_months (0.130) is the strongest predictor of churn, followed closely by csat_score (0.124) and monthly_logins (0.114). This suggests that churn risk is highest among newer, less engaged, and less satisfied customers. Operational factors such as payment_failures (0.077) and avg_resolution_time (0.054) also contribute meaningfully to churn behavior.
Permutation importance is used in KNeighborsClassifier to verify feature importance. The model finds the features csat_score, discount_applied, and payment_failures to be highly important. Feature importance is inferred where one feature is shuffled randomly at a time. The change in model performance is then measured - the drop in performance, for example, is inferred as the magnitude of feature importance. csat_score  (0.011) has the largest drop in performance and is, therefore, the most important feature - customers who leave don't leave randomly but because they are unhappy; unhappiness precedes churn. discount_applied  (0.008)  customers might be price sensitive; churn happens when promotional periods end; the high importance means that the presence or absence of a discount is a major factor in a customer's decision to stay or not to stay. payment_failures  (0.005) - this is sometimes known as involuntary churn; if a card expires or a bank rejects a payment, some customers don't bother to update their info - they just let the account die.
The SVC (Support Vector Classifier)model finds the features csat_score, monthly_logins, and payment_failures to be highly important. csat_score (0.074) is the most important feature and is the strongest predictor of churn. Lower score increases the churn probability. In order to reduce churn, satisfaction should be improved. monthly_logins (0.031) is the second most important feature. Customers who log in less frequently are more likely to churn. Customers with high engagement have low churn risk. payment_failures (0.030) is almost as important as engagement. Customers experiencing payment issues are at a higher risk of churn. Billing friction or financial dissatisfaction could be an issue.

Overall Model Summary     

The DecisionTreeClassifier model is the clear winner. For this particular dataset, it is the recommended machine learning algorithm. The model has the highest profit, Accuracy, Precision, F2 Score and AUC. While it is last in terms of Recall, this is more than made up by its higher Accuracy and Precision.
While the DecisionTreeClassifier model provided the most balanced performance with an Accuracy of 74.8% and the highest Profit, some of the other models achieved higher Recall (>96%), ensuring that almost all potential churners are identified. However, this comes at the cost of False Positives, as evidenced by low Precision.

Next Steps and Recommendations  

- Confirm the model that will suit the business needs in terms of the optimal level of churn identification and precision.    
- Continue model development to include actual identification of clients who are likely to churn.
- Tune the thresholds to maximize profit using realistic lifetime value and cost of retention assumptions.
- Deploy and apply the model for the use of relevant business groups like marketing teams for retention campaigns, customer support teams for proactive outreach, and product teams for feature improvements. Leadership can also be guided by the model for churn strategy and forecasting.     
- Continue model development to validate the features relative importance to guide management on which features need to be given particular attention in order to prevent churn.
Churn in this dataset is primarily driven by customer satisfaction and engagement levels rather than pricing. Improving user experience and increasing product adoption would likely have the strongest impact on reducing churn.

Reference: Jupyter Notebook 
           Customer_Churn_Prediction.ipynb
           Ronaldo Bantayan (Author)
