class ->
MyPreprocessor
MyLinearRegession
MyLogisticRegression
Kfold

for fitting the model in linear regression,
use 
'model.fit(X,y)'
to predict,
'model.predict(Xtest)'

Default parameters, 
train under RMSE
learning rate=0.01 (use 'model.alpha=input' to change rate)
epochs=150    (use 'model.iter=input' to change epochs)

for kfold on linear regression,

use 'Kfold(K,ErrorName,model (Linear regression),y,error)'
then 'Kfold.proc()' to split the data into k folds
then 'use Kfold.eval()' to get the plot the cost of best of the fold



for fitting the model in logistic regression,
use
Train using BGD
'model.fit(X,y)'
to predict,
'model.predict(Xtest)'

use threefit(train_x,train_y,validate_x,validate_y,test_x,test_y,gradient name) 
to get the plot of loss of train and val set

Default parameters,
learning rate=0.1 (use 'model.alpha=input' to change rate)
epochs=150    (use 'model.iter=input' to change epochs)

skilearn code is done in skilearn.py
just run the code

EDA is done on eda.py
just run the code

