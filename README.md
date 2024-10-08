# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2024-10-08 105540](https://github.com/user-attachments/assets/002a6ae9-db4a-4461-a318-550e1a0d9ac2)
```
df.dropna()
```
![Screenshot 2024-10-08 105616](https://github.com/user-attachments/assets/e4ac4253-85f9-4c80-8997-bfe6acc95412)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-10-08 105710](https://github.com/user-attachments/assets/c5944b22-8d5e-4e54-a062-c5503818db26)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-08 105758](https://github.com/user-attachments/assets/9bef9d02-25c2-4d0c-9f1d-b646ba674aaa)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-08 105832](https://github.com/user-attachments/assets/353f6035-e1c1-4ce4-b2c5-a91d428b3f2d)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-08 105914](https://github.com/user-attachments/assets/47826701-13eb-4709-8a75-5800613891a5)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-08 105939](https://github.com/user-attachments/assets/a8e803f4-cd9b-47ea-bcbe-d741a34a953b)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-10-08 110012](https://github.com/user-attachments/assets/9b2c4418-fee3-4290-953f-3fe39bedadab)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![Screenshot 2024-10-08 110105](https://github.com/user-attachments/assets/7a98425d-4e74-4f79-8dc2-067ed61b9d62)
```
data.isnull().sum()
```
![Screenshot 2024-10-08 110133](https://github.com/user-attachments/assets/313914f8-8e52-4ec3-8304-bcd772ac2069)
```
missing = data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-08 110230](https://github.com/user-attachments/assets/4e7e3f61-baf5-4834-9726-5c09df459a7d)
```
data2 = data.dropna(axis=0)
data2
```
![Screenshot 2024-10-08 110314](https://github.com/user-attachments/assets/cf053484-e7f8-4085-bff2-4a3c66b5e20f)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-08 110433](https://github.com/user-attachments/assets/0e6a92a3-10d4-4ba9-a5d3-ff1f19caccd8)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-08 110523](https://github.com/user-attachments/assets/d90e9b01-8cc6-4d29-92a6-568ea1f63b0c)
```
data2
```
![Screenshot 2024-10-08 110551](https://github.com/user-attachments/assets/97b6b9f7-768e-4c7d-90e0-f51f317ff186)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![Screenshot 2024-10-08 110649](https://github.com/user-attachments/assets/72aa6752-6969-46f9-8d20-e5b16d6d3467)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-08 110739](https://github.com/user-attachments/assets/95477dc6-6f5f-4848-a3b7-7adea1daceb9)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-08 110811](https://github.com/user-attachments/assets/3ee37355-13d2-4080-a7e2-bcea882eb12b)
```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-08 110842](https://github.com/user-attachments/assets/497da412-fd76-4289-b8f3-cc12fa6db57d)
```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-08 110920](https://github.com/user-attachments/assets/7b359b89-5bd6-461f-9f88-95eed2f4c153)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-08 110955](https://github.com/user-attachments/assets/fa7e5bce-a954-4d0c-bf6b-f2e93d9d198c)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-08 111041](https://github.com/user-attachments/assets/c98a9f99-076b-491e-9618-4642977ac80b)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-08 111108](https://github.com/user-attachments/assets/0698f619-575d-4e82-baaa-73564d9f2d1f)
```
print('Misclassified samples: %d'%(test_y!=prediction).sum())
```
![Screenshot 2024-10-08 111131](https://github.com/user-attachments/assets/d4bade4e-a1d2-4ef5-9d9d-e63f551f994e)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-10-08 111209](https://github.com/user-attachments/assets/8c472f81-3915-45ec-9712-aa733af31773)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-08 111234](https://github.com/user-attachments/assets/a8401430-89ac-4c86-ab4f-7fa08b58032a)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
```
![Screenshot 2024-10-08 111255](https://github.com/user-attachments/assets/dea3230b-dd3d-4625-85de-27768c57966b)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![Screenshot 2024-10-08 111337](https://github.com/user-attachments/assets/c4a2da37-7ad1-4558-bbfc-4e1afd5b1e3f)
```
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-10-08 111417](https://github.com/user-attachments/assets/9cbeee5e-763f-4f96-926e-9fc2c908735d)

# RESULT:
Thus the code for  Feature Scaling and Feature Selection process has been executed.
