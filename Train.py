import pandas as pd, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:\\Users\\ashish\\Downloads\\ml_student_placement_100_rows.csv")
x, y = df[['IQ', 'CGPA']], df['Placement']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

logic = LogisticRegression().fit(x_train, y_train)
print(accuracy_score(y_test, logic.predict(x_test)))

pickle.dump(logic, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))