from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)

# Create your k-fold splits or train-val-test splits as required



linear = MyLinearRegression()
linear.fit(X, y)

ypred = linear.predict(X)

print('Predicted Values:', ypred)
print('True Values:', y)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)

# Create your k-fold splits or train-val-test splits as required



logistic = MyLogisticRegression()
logistic.fit(X, y)

ypred = logistic.predict(X)

print('Predicted Values:', ypred)
print('True Values:', y)

    
'''
allfold=[]
x=MyPreProcessor()
X,y=x.pre_process(0)
x=MyLinearRegression()

for i in range(3,6):
    Y=Kfold(i,x,X,y,"MAE")
    Y.proc()
    costtest=Y.eval()
    l=[]
    for j in range(i):
        l.append(costtest[j][-1])
    s=sum(l)/len(l)
    allfold.append(s)                        #for Kfold v/s error graph in mae
    
plt.scatter([3,4,5],[allfold])
plt.title("MAE")
plt.ylabel('Loss')
plt.xlabel('K')
plt.show()'''



'''Y=Kfold(3,x,X,y,"MAE")
Y.proc()                                #for Kfold plots, change the error to get rmse plot
Y.eval()   '''    
    
    
    




'''x=MyPreProcessor()
X,y=x.pre_process(0)


x=MyLinearRegression()
x.iter=300
thetas=x.normal_equation(X,y)

print(thetas)


print(x.cost_functionMAE(X,y,thetas))             #code for finding cost from parameters of cost function of dataset1

Y=Kfold(3,x,X,y,"MAE")
Y.proc()
Xtrain,ytrain,Xtest,ytest=Y.eval()

print(x.cost_functionMAE(Xtrain,ytrain,thetas))

print(x.cost_functionMAE(Xtest,ytest,thetas))'''




'''
x=MyPreProcessor()              
X,y=x.pre_process(2)
train_x, validate_x, test_x = np.split(X, [int(.7 * len(X)), int(.8 * len(X))])      #train val test split for dataset 2
train_y, validate_y, test_y = np.split(y, [int(.7 * len(y)), int(.8 * len(y))])

x=MyLogisticRegression()

x.threefit(train_x,train_y,validate_x,validate_y,test_x,test_y,"SGD")            #Performing fit and this will plor graph of loss for both train and val set
#Default learning rate=1, epochs=10000

ans=x.predict(test_x)                            

print(np.count_nonzero(ans==test_y)/len(ans)*100) '''      #accuracy 
