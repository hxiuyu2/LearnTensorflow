import pandas as pd
census = pd.read_csv('census_data.csv')

# convert label to 0 or 1
# find out possible values for each categorical column,
# TypeError: expected bytes, int found
# values = pd.DataFrame(census['income_bracket'].unique())
# for index,row in values.iterrows():
#     census.loc[census['income_bracket'] == row[0],'income_bracket'] = index
def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1
census['income_bracket'] = census['income_bracket'].apply(label_fix)

# Train Test split
from sklearn.model_selection import train_test_split
# drop: dont forget axis = 1
x_data = census.drop('income_bracket', axis=1)
labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.3,random_state=101)

# create features
import tensorflow as tf
cate_column = ['workclass', 'education', 'occupation', 'relationship','race','gender','native_country','marital_status']
num_column = ['age','education_num','capital_gain','capital_loss','hours_per_week']
feature_cols = []
for name in cate_column:
    feature_cols.append(tf.feature_column.categorical_column_with_hash_bucket(name, census[name].nunique()))
for name in num_column:
    feature_cols.append(tf.feature_column.numeric_column(name))

# # define input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feature_cols)
model.train(input_fn=input_func, steps=20000)

# evaluate model
pred_input = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(model.predict(input_fn=pred_input))
final_pred = [pred['class_ids'][0] for pred in predictions]
from sklearn.metrics import classification_report
print("Linear Classification:")
print(classification_report(y_test,final_pred))

# change the model to DNN
feature_cols = []
# embeded feature for categorical in DNN
for name in cate_column:
    nunique = census[name].nunique()
    cate_col = tf.feature_column.categorical_column_with_hash_bucket(name, nunique)
    feature_cols.append(tf.feature_column.embedding_column(cate_col,nunique))
for name in num_column:
    feature_cols.append(tf.feature_column.numeric_column(name))
dnn = tf.estimator.DNNClassifier(hidden_units=[10,10,6],feature_columns=feature_cols,n_classes=2)
dnn.train(input_func,steps=20000)
pred_input = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(dnn.predict(input_fn=pred_input))
final_pred = [pred['class_ids'][0] for pred in predictions]
print("NN Classification:")
print(classification_report(y_test,final_pred))