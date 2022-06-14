import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle


df=pd.read_csv('titanic.csv')
df.Age= df.Age.fillna(df.Age.median())
le_Sex=LabelEncoder()
le_Embarked=LabelEncoder()
df['Sex_n']=le_Sex.fit_transform(df['Sex'])

df.Embarked= df.Embarked.fillna('S')
df['Embarked_n']=le_Embarked.fit_transform(df['Embarked'])

x=df.drop(['Survived','PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
y=df.Survived

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3)

model=tree.DecisionTreeClassifier()
model.fit(x,y)
pickle.dump(model,open('model.pkl','wb'))

# y_pred=model.predict(x_test)
# print(y_pred)