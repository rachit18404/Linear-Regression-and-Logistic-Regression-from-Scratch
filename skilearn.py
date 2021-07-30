from scratch import MyPreProcessor,MyLogisticRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
x=MyPreProcessor()
X,y=x.pre_process(2)
train_x, validate_x, test_x = np.split(X, [int(.7 * len(X)), int(.8 * len(X))])
train_y, validate_y, test_y = np.split(y, [int(.7 * len(y)), int(.8 * len(y))])

x=MyLogisticRegression()


x.threefit(train_x,train_y,validate_x,validate_y,test_x,test_y,"SGD")

ans=x.predict(test_x)

print(np.count_nonzero(ans==test_y)/len(ans)*100)




model = LogisticRegression()     
model.fit(train_x,train_y)            #fit the training set  from logistic regression     
print(model.score(train_x,train_y))              #accuracy of train
print(model.score(test_x,test_y))                      #accuracy of test
#print(model.predict(train_x))    
