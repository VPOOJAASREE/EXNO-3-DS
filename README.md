## EXNO-3-DS

NAME V. POOJAA SREE

REG.: 212223040147

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df.head()
```

# OUTPUT:

![1](https://github.com/user-attachments/assets/cf23961f-2fca-4e36-b660-47122f3d123e)


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

# OUTPUT:

![2](https://github.com/user-attachments/assets/d1326ce5-9c8f-4d96-ab48-275734255975)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```

# OUTPUT:

![3](https://github.com/user-attachments/assets/7014c061-e462-4288-8a5e-7981dd5ee2da)



```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```

# OUTPUT:

![4](https://github.com/user-attachments/assets/28254ad9-fe46-4ac4-984c-1c3fb50169ef)



```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

```


```
df2=pd.concat([df2,enc],axis=1)
df2

```

# OUTPUT:

![6](https://github.com/user-attachments/assets/7291bead-6d93-4e77-ba55-8bec37707620)



```
pd.get_dummies(df2,columns=["nom_0"])

```

# OUTPUT:

![7](https://github.com/user-attachments/assets/a71d7d20-758b-43a0-8886-eef84971525e)



```
pip install --upgrade category_encoders

```

# OUTPUT:

![8](https://github.com/user-attachments/assets/a85c45fa-0f2e-4680-8d68-8421e4acd3e3)


```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

```

# OUTPUT:

![9](https://github.com/user-attachments/assets/0466f24f-2625-495d-a40e-ad388d35a703)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

```

# OUTPUT:

![10](https://github.com/user-attachments/assets/7104ee9b-befb-4dc7-a265-318a8a995971)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```

# OUTPUT:

![11](https://github.com/user-attachments/assets/d574133b-5048-4ff0-8b6f-c1e1d00ef501)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

```

# OUTPUT:

![12](https://github.com/user-attachments/assets/f52ddb5e-8251-4550-848d-3a88312e608a)


```
df.skew()

```

# OUTPUT:

![13](https://github.com/user-attachments/assets/e1ad6d51-1b4f-446d-a7aa-3c3c2665b759)


```
np.log(df["Highly Positive Skew"])

```

# OUTPUT:

![14](https://github.com/user-attachments/assets/84ebaec0-6428-4da3-92f9-5ae2eb533365)


```
np.reciprocal(df["Moderate Positive Skew"])

```

# OUTPUT:

![15](https://github.com/user-attachments/assets/8ed7594a-b9bd-4d94-a500-18fb72eea27f)


```
np.sqrt(df["Highly Positive Skew"])

```

# OUTPUT:

![16](https://github.com/user-attachments/assets/be7dda88-b875-4a39-bd69-c1435395dc63)


```
np.square(df["Highly Positive Skew"])

```

# OUTPUT:

![17](https://github.com/user-attachments/assets/a49ce1b8-f328-4886-9272-12cf9578d3d3)


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```

# OUTPUT:

![18](https://github.com/user-attachments/assets/8d88513d-7b27-47d5-886f-e7575d06f543)


```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])

```

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

# OUTPUT:

![20](https://github.com/user-attachments/assets/73eac225-3f84-4654-9ccd-cc4db217c734)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```

# OUTPUT:

![21](https://github.com/user-attachments/assets/b3e63284-b46e-4419-8608-eb534ffd6d19)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```

# OUTPUT:

![22](https://github.com/user-attachments/assets/714a7d3c-6118-4dfe-8ab8-b9aefa834380)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

# OUTPUT:

![23](https://github.com/user-attachments/assets/18f3daf2-4fd2-4068-9f22-321cc94e2e3c)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

# OUTPUT:

![24](https://github.com/user-attachments/assets/d44ba7e6-34fc-42ad-bea4-ba0191ef5d6e)



# RESULT:
      ```
           Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

      ```
       
