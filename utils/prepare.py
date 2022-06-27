from sklearn.preprocessing import LabelEncoder

def prepare(dataset):
  dataset['RainTomorrow'] = dataset['RainTomorrow'].map({'Yes': 1, 'No': 0})
  dataset['RainToday'] = dataset['RainToday'].map({'Yes': 1, 'No': 0})

  # Filling the missing values for continuous variables with mode
  dataset['MinTemp']=dataset['MinTemp'].fillna(dataset['MinTemp'].mean())
  dataset['MaxTemp']=dataset['MinTemp'].fillna(dataset['MaxTemp'].mean())
  dataset['Rainfall']=dataset['Rainfall'].fillna(dataset['Rainfall'].mean())
  dataset['Evaporation']=dataset['Evaporation'].fillna(dataset['Evaporation'].mean())
  dataset['Sunshine']=dataset['Sunshine'].fillna(dataset['Sunshine'].mean())
  dataset['WindGustSpeed']=dataset['WindGustSpeed'].fillna(dataset['WindGustSpeed'].mean())
  dataset['WindSpeed9am']=dataset['WindSpeed9am'].fillna(dataset['WindSpeed9am'].mean())
  dataset['WindSpeed3pm']=dataset['WindSpeed3pm'].fillna(dataset['WindSpeed3pm'].mean())
  dataset['Humidity9am']=dataset['Humidity9am'].fillna(dataset['Humidity9am'].mean())
  dataset['Humidity3pm']=dataset['Humidity3pm'].fillna(dataset['Humidity3pm'].mean())
  dataset['Pressure9am']=dataset['Pressure9am'].fillna(dataset['Pressure9am'].mean())
  dataset['Pressure3pm']=dataset['Pressure3pm'].fillna(dataset['Pressure3pm'].mean())
  dataset['Cloud9am']=dataset['Cloud9am'].fillna(dataset['Cloud9am'].mean())
  dataset['Cloud3pm']=dataset['Cloud3pm'].fillna(dataset['Cloud3pm'].mean())
  dataset['Temp9am']=dataset['Temp9am'].fillna(dataset['Temp9am'].mean())
  dataset['Temp3pm']=dataset['Temp3pm'].fillna(dataset['Temp3pm'].mean())
  dataset['RainToday']=dataset['RainToday'].fillna(dataset['RainToday'].mode()[0])
  dataset['RainTomorrow']=dataset['RainTomorrow'].fillna(dataset['RainTomorrow'].mode()[0])
  dataset['WindDir9am'] = dataset['WindDir9am'].fillna(dataset['WindDir9am'].mode()[0])
  dataset['WindGustDir'] = dataset['WindGustDir'].fillna(dataset['WindGustDir'].mode()[0])
  dataset['WindDir3pm'] = dataset['WindDir3pm'].fillna(dataset['WindDir3pm'].mode()[0])

  # Encoding the categorical variables
  le = LabelEncoder()
  dataset['Location'] = le.fit_transform(dataset['Location'])
  dataset['WindDir9am'] = le.fit_transform(dataset['WindDir9am'])
  dataset['WindDir3pm'] = le.fit_transform(dataset['WindDir3pm'])
  dataset['WindGustDir'] = le.fit_transform(dataset['WindGustDir'])

  # Drop highly correlated columns
  dataset = dataset.drop(['Temp3pm', 'Temp9am', 'Humidity9am'],axis=1)

  # Drop date column
  dataset=dataset.drop(['Date'],axis=1)

  return dataset
