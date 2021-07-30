import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        self.thetasMAE=[]
        self.thetasRMSE=[]
        pass

    def pre_process(self, dataset):


       
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            train_df=pd.read_csv("Dataset.csv")
            train_df = train_df.drop_duplicates(keep='last')
            train_df['c0'] = train_df['c0'].map({'M':2,'F':1,'I':1}) #Assigning gender to integer values
            train_df=train_df.loc[:,["c0","c1","c2","c3","c4","c5","c6","c7","c8"]].apply(pd.to_numeric,errors='coerce')
            train=train_df.loc[:,["c0","c1","c2","c3","c4","c5","c6","c7"]]
            test=train_df.loc[:,["c8"]]
            
            min_val=train["c0"].min()                                #Scaling gender
            max_val=train["c0"].max() 
            train['c0'] = train['c0'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))


            '''         min_val=train["c1"].min()
            max_val=train["c1"].max()
            train['c1'] = train['c1'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            
            min_val=train["c2"].min()
            max_val=train["c2"].max()
            train['c2'] = train['c2'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))
            
            min_val=train["c3"].min()
            max_val=train["c3"].max()
            train['c3'] = train['c3'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            min_val=train["c4"].min()
            max_val=train["c4"].max()
            train['c4'] = train['c4'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            min_val=train["c5"].min()
            max_val=train["c5"].max()
            train['c5'] = train['c5'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))


            min_val=train["c6"].min()
            max_val=train["c6"].max()
            train['c6'] = train['c6'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            min_val=train["c7"].min()
            max_val=train["c7"].max()
            train['c7'] = train['c7'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))'''

            X=np.asarray(train)                                        #getting feature set and output set as array
            y=np.asarray(test)
    
            pass
        elif dataset == 1:
            # Implement for the video game dataset

            train_df=pd.read_csv("VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv")
            train_df = train_df.sample(frac = 1)                        #Shuffling Data
            train_df = train_df[train_df['Global_Sales'] < 20]          #Removing outliers
            feature_train=train_df.loc[:,["Critic_Score","User_Score","Global_Sales"]]        #feature selection
            

            feature_train=feature_train.loc[:,["Critic_Score","User_Score","Global_Sales"]].apply(pd.to_numeric,errors='coerce')  #converting string to float
            feature_train=feature_train.dropna(axis=0)
            #print(feature_train)
            #feature_train=feature_train.fillna(feature_train.median())
            train=feature_train.loc[:,["Critic_Score","User_Score"]]           #Seperating features with output value
            test=feature_train.loc[:,["Global_Sales"]]
            #train=train.fillna(train.median())
            min_val=feature_train["User_Score"].min()
            max_val=feature_train["User_Score"].max()                           #Scaling data
            #print(min_val,max_val)
            train['User_Score'] = train['User_Score'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))
            min_val=feature_train["Critic_Score"].min()
            max_val=feature_train["Critic_Score"].max()                     
            #print(min_val,max_val)
            train["Critic_Score"] = train["Critic_Score"].apply(lambda x: (((x - min_val)/(max_val - min_val) )))
            #print(feature_train["Critic_Score"])
            train=np.asarray(train)                                            #converting dataframe to array
            test=np.asarray(test)
            #print(train)
            #print(len(train),len(test))
            X=train
            y=test
            #print(type(feature_train))
            pass
        elif dataset == 2:
            train_df=pd.read_csv("data_banknote_authentication.csv")                  
            train_df = train_df.sample(frac = 1)               
            
            train_df=train_df.loc[:,["c1","c2","c3","c4","c5"]].apply(pd.to_numeric,errors='coerce')    #converting string to float
            train=train_df.loc[:,["c1","c2","c3","c4"]]                                        #feature selection
            test=train_df.loc[:,["c5"]]
            
            min_val=train["c1"].min()
            max_val=train["c1"].max()
            train['c1'] = train['c1'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))              #feature scaling

            min_val=train["c2"].min()
            max_val=train["c2"].max()
            train['c2'] = train['c2'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            min_val=train["c3"].min()
            max_val=train["c3"].max()
            train['c3'] = train['c3'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))

            min_val=train["c4"].min()
            max_val=train["c4"].max()
            train['c4'] = train['c4'].apply(lambda x: (((x - min_val)/(max_val - min_val) )))


            
            X=np.asarray(train)
            #y=np.asarray(test)
            
            y=test['c5'].values
           
            
            pass

        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """
    thetas=[]
    costhistoryRMSE=[]
    costhistoryRMSEtest=[]
    costhistoryMAE=[]
    costhistoryMAEtest=[]
    Xtest=[]
    ytest=[]
    thetasMAE=[]
    thetasRMSE=[]
    alpha=0.01
    iter=150
    def __init__(self):
        
        pass

    

    def updateThetaMAE(self,train,test,thetas,alpha):                       #Gradient for MSE which updates theta by subtracting theta with theta_der 
        theta_der=[]
        users=len(train)

        
       
        for k in range(len(thetas)):
            loss=0
        
            for i in range(users):
                s=0                                     
                for j in range(len(thetas)):
                    if j==0:
                        s+=thetas[j]
                        
                        
                    else:
                        s+=(thetas[j]*train[i][j-1])
          
                hx=s-test[i]
                if hx>0:
                    hx=1
                elif hx<0:
                    hx=-1
                else:
                    hx=0
                
                
                if k==0:
                    pass
                else:
                 #   print(hx,"*",train[i][k-1])
                    hx=hx*train[i][k-1]
                  #  print(hx)
                
                    
                
                
                loss+=hx
                #print("loss",loss)

            theta_der.append(loss/users)
           # print("loss",loss)
            
        theta_der=alpha*np.array(theta_der)
        
        #print(thetas)
        
        theta=[x1 - x2 for (x1, x2) in zip(thetas, theta_der)]
        
        #print(theta)
        return(theta)

            




    def updateThetaRMSE(self,train,test,thetas,alpha):
        theta_der=[]
        users=len(train)
        for k in range(len(thetas)):
            loss=0                                                                        #Gradient for RMSE which updates theta by subtracting theta with theta_der 
            hxsquare=0
            for i in range(users):
                s=0
                for j in range(len(thetas)):
                    if j==0:
                        s+=thetas[j]
                 #       print("theta bias:", s)
                        
                    else:
                        s+=(thetas[j]*train[i][j-1])
                  #      print("x :", j-1, train[i][j-1],thetas[j]*train[i][j-1],s)
                    
                hx=s-test[i]
                #print(s,test[i],hx)
                #print("hxsquare",math.pow(hx,2))
                hxsquare+=math.pow(hx,2)
                if k==0:
                    #print("hx",hx)
                    pass
                else:
                   # print(hx,"*",train[i][k-1])
                    hx=hx*train[i][k-1]
                    #print(hx)
                    
                
                
                loss+=hx
                #print("loss",loss)
              
            #print(loss,k)
            num=loss/users
            #print(num, k)
            #print(hxsquare)
            hxsquare=hxsquare/(2*users)
            Jtheta=math.sqrt(hxsquare)
            #print("Jtheta, num",2*Jtheta,num)
            #print("num",num)
            td=num/(2*Jtheta)
            td=td[0]
            #print(td)
            theta_der.append(td)
        


        #print("theeeee",theta_der)


        theta_der=alpha*np.array(theta_der)
        
        #print(thetas)
        
        theta=[x1 - x2 for (x1, x2) in zip(thetas, theta_der)]
        
        #print(theta)
        return(theta)
    
    
        """
        Fitting (training) the linear model.

        

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self


        """
    '''cost=1/2m sum(ypred-y)**2'''
    def cost_functionRMSE(self,train,test,thetas):
            
           # print(thetas)
            users=len(train)
            loss=0
            for i in range(users):
                s=0
            #    print(s)
                for j in range(len(thetas)):
                    if j==0:
                        s=s+thetas[j]
             #           print(thetas[j])
                    else:
                        s+=(thetas[j]*train[i][j-1])
              #          print(thetas[j])
                hx=s-test[i]
               # print(test[i],s)
                loss+=math.pow(hx,2)
                
            return math.sqrt(loss/(2*users))

    """cost= 1/m sum(ypred-y)''' '''ypred=thetaT * Xi"""            
    def cost_functionMAE(self,train,test,thetas):
            
            users=len(train)
            loss=0
            for i in range(users):
                s=0
                for j in range(len(thetas)):
                    if j==0:
                        s+=thetas[j]
                        
                    else:
                        s+=thetas[j]*train[i][j-1]                #Cost function for MAE 
                        
    
                
                hx=abs(s-test[i])                          
                
                loss+=hx
                
                #print("loss",loss,i)
                
                
            return float(loss/(users)) 

    def trainModelRMSE(self,train,test,thetas,alpha,iters):
        cost_history = []
        for i in range(iters):
            thetas=self.updateThetaRMSE(train,test,thetas,alpha)
            #print(thetas)
        
            cost=self.cost_functionRMSE(train,test,thetas)                        # it updates cost history and theta by calling cost history and update_theta function of RMSE
            #print(cost)
            cost_history.append(cost)                                          
        itr=[i for i in range(iters)]
        #plt.plot(itr,cost_history)
        #plt.show()
        
        #print(cost_history)
        self.thetas=thetas
        return cost_history

    def trainModelMAE(self,train,test,thetas,alpha,iters):
        cost_history = []
        for i in range(iters):
            thetas=self.updateThetaMAE(train,test,thetas,alpha)          # it updates cost history and theta by calling cost history and update_theta function of MAE
            #print(thetas)
        
            cost=self.cost_functionMAE(train,test,thetas)
            #print(cost)
            cost_history.append(cost)
        itr=[i for i in range(iters)]
        #plt.plot(itr,cost_history)
        #plt.show()
        
        #print(cost_history)
        self.thetas=thetas
        return cost_history







    def ktrainModelRMSE(self,train,test,thetas,alpha,iters):
        cost_history = []
        costhistoryRMSEtest=[]
        for i in range(iters):
            thetas=self.updateThetaRMSE(train,test,thetas,alpha)
           
        
            cost=self.cost_functionRMSE(train,test,thetas)              # it updates cost history and theta by calling cost history and update_theta function of RMSE for train and test plot
            costt=self.cost_functionRMSE(self.Xtest,self.ytest,thetas)
            
            cost_history.append(cost)
            costhistoryRMSEtest.append(costt)
            
        itr=[i for i in range(iters)]
    

        self.thetasRMSE=thetas
        self.costhistoryRMSEtest=costhistoryRMSEtest
       
        return cost_history

    def ktrainModelMAE(self,train,test,thetas,alpha,iters):
        cost_history = []
        costhistoryMAEtest=[]
        for i in range(iters):
            thetas=self.updateThetaMAE(train,test,thetas,alpha)   # it updates cost history and theta by calling cost history and update_theta function of RMSE for train and test plot
            
            cost=self.cost_functionMAE(train,test,thetas)
            costt=self.cost_functionMAE(self.Xtest,self.ytest,thetas)
            
            cost_history.append(cost)
            costhistoryMAEtest.append(costt)
         
      
        self.thetasMAE=thetas
        self.costhistoryMAEtest=costhistoryMAEtest
        return cost_history
    

    def fit(self, train, test):
        self.thetas=[0]*(len(train[0])+1)
        self.costhistoryRMSE=self.trainModelRMSE(train,test,self.thetas,self.alpha,self.iter)     
    
        return self

   

    def kfit(self,X,y,Xtest,ytest,error):
        self.Xtest=Xtest
        self.ytest=ytest
        if error=="RMSE":
                    self.thetasRMSE=[0]*(len(X[0])+1)
                    self.costhistoryRMSE=self.ktrainModelRMSE(X,y,self.thetasRMSE,self.alpha,self.iter)     #fitting the model here which call for trainmodel
        elif error=="MAE":
                    self.thetasMAE=np.asarray([0.0]*(len(X[0])+1))
                    
                    self.costhistoryMAE=self.ktrainModelMAE(X,y,self.thetasMAE,self.alpha,self.iter)            
   
        
        return self   
    def predict(self, train):
        out=[]
        idx=[i for i in range(len(train))]
        for i in range(len(train)):
            ans=0
            for j in range(len(self.thetas)):
                if j==0:
                    ans+=self.thetas[j]
                else:
                    ans+=self.thetas[j]*train[i][j-1]             #return the predicted values based on the theta from training model

            out.append(ans)
        
        #plt.scatter(idx,out)
        #plt.show()
       
  
        # return the numpy array y which contains the predicted values
        return out
    
    def normal_equation(self,X,y):
        addcolumn=np.asarray([[1] for i  in range(len(X))])        #normal equation for best theta
        
        Xnew=np.column_stack((addcolumn,X))
        oo=np.linalg.inv(Xnew.T.dot(Xnew)).dot(Xnew.T).dot(y)
        
        self.thetas=oo
        return oo
        


class MyLogisticRegression():
    Xval=[]
    yval=[]
    Xtest=[]
    ytest=[]
    costhistoryval=[]
    costhistorytest=[]
    thetas=[]
    bias=0
    alpha=1   
    iter=10000    
    def __init__(self):

        pass

    def update_theta(self,X,y,thetas,bias,alpha):
        #print(thetas.T)
        thetaTx=np.matmul(X,thetas.T)+bias
        #print(thetaTx)                                    

            
        ex=np.exp(-1*thetaTx)
        den=1+ex
        ypred=np.reciprocal(den)                         #Gradient which updates theta by subtracting theta with theta_der
                                                         # theta=theta- alpha dtheta
       
        #print(ypred[:60])                              # dtheta=(yhat-y)*x/m
        #print(y)
        col=ypred-y
        #print(col)
        m=y.shape[0]
    
        theta_der=np.matmul(col.T,X)/m
        dbias=np.sum(col)/m
        
        theta_der=theta_der*alpha
        thetas=thetas-theta_der
        
        dbias=dbias*alpha
        bias=bias-dbias
        
        #print(thetas,bias)
        return thetas,bias





    def cost_function(self,X,y,thetas,bias):
        thetaTx=np.matmul(X,thetas.T)+bias
        
        ex=np.exp(-1*thetaTx)
        den=1+ex
        hx=np.reciprocal(den)

        left=np.log(hx)
        left=y*left                          #implementing cost function 


        r1=1-y
        r2=1-hx
        r2=np.log(r2)
        right=r1*r2
        s=left+right
        #print(len(s))
        s=np.sum(s)
        Jtheta=s/len(X)
        Jtheta=-1*Jtheta
        #print(Jtheta)
        return Jtheta







    def trainModel(self,train,test,thetas,bias,alpha,iters):
        cost_history = []
        for i in range(iters):
            thetas,bias=self.update_theta(train,test,thetas,bias,alpha)
            #print(thetas)
        
            cost=self.cost_function(train,test,thetas,bias)
            #print(cost)
            cost_history.append(cost)

        itr=[i for i in range(iters)]
        #plt.plot(itr,cost_history)
        #plt.show()
        
        #print(cost_history)
        self.thetas=thetas
        self.bias=bias
        return cost_history       

    def trainModelthree(self,train,test,thetas,bias,alpha,iters):
        cost_historyval = []
        costhistorytest=[]
        for i in range(iters):
            thetas,bias=self.update_theta(train,test,thetas,bias,alpha)     #training model SGD, one iteration here only becaue only 1 row is taken and it's thetas has been used in finding loss for val and test
            
        
            cost=self.cost_function(self.Xval,self.yval,thetas,bias)            
            cost_historyval.append(cost)

            
            costt=self.cost_function(train,test,thetas,bias)
            costhistorytest.append(costt)
            

        itr=[i for i in range(iters)]
        #plt.plot(itr,cost_history)
        #plt.show()
        
        #print(cost_history)
        self.thetas=thetas
        self.bias=bias
        self.costhistorytest=costhistorytest
        return cost_historyval


    def trainModelthreeSGD(self,train,test,thetas,bias,alpha,iters):
        #print(self.costhistoryval)
        
        for i in range(iters):
            thetas,bias=self.update_theta(train,test,thetas,bias,alpha)#training model SGD, one iteration here only becaue only 1 row is taken and it's thetas has been used in finding loss for val and tes
        
            cost=self.cost_function(self.Xval,self.yval,thetas,bias)            
            self.costhistoryval.append(cost)

            
            costt=self.cost_function(self.Xtest,self.ytest,thetas,bias)
            self.costhistorytest.append(costt)
            

        itr=[i for i in range(iters)]
        #plt.plot(itr,cost_history)
        #plt.show()
        
        #print(cost_history)
        self.thetas=thetas
        self.bias=bias
        
        return self 
        
    

    def fit(self, X, y):

        self.thetas = np.zeros(X.shape[1])
        self.bias=0
        self.costhistory=self.trainModel(X,y,self.thetas,self.bias,self.alpha,self.iter)
        return self


    def threefit(self,Xtrain,ytrain,Xval,yval,Xtest,ytest,mode):
        self.Xval=Xval
        self.yval=yval
        self.Xtest=Xtest
        self.ytest=ytest
        if mode=="BGD":
            self.thetas = np.zeros(Xtrain.shape[1])
            self.bias=0
            self.costhistoryval=self.trainModelthree(Xtrain,ytrain,self.thetas,self.bias,self.alpha,self.iter)   # Differet fit() for train val test set to plot loss of train an val
            
            plt.plot(self.costhistoryval)
            plt.plot(self.costhistorytest)
            plt.show()
            return self
        if mode=="SGD":
            self.thetas = np.zeros(Xtrain.shape[1])
            self.bias=0
            self.costhistoryval=[]
            self.costhistorytest=[]
            import random
            for i in range(self.iter):
                b=random.randint(1,int(Xtrain.shape[0]-1))
                
                XXX=Xtrain[b].tolist()
                yyy=ytrain[b].tolist()            
                self.trainModelthreeSGD(np.asarray([XXX]),np.asarray([yyy]),self.thetas,self.bias,self.alpha,1)
            
            plt.plot(self.costhistoryval)
            plt.plot(self.costhistorytest)
            plt.show()
            return self
            
            

        
            
        
        

    def predict(self, X):
        thetaTx=np.matmul(X,self.thetas.T)+self.bias
        
        ex=np.exp(-1*thetaTx)
        den=1+ex
        hx=np.reciprocal(den)
        hx=np.round_(hx)

        # return the numpy array y which contains the predicted values
        return hx
class Kfold:

    alltrain=[]
    alltest=[]
    testpredict=[]
    testout=[]
    
    def __init__(self,k,model,train,test,error):
        self.k=k
        self.model=model
        self.train=train
        self.test=test
        self.error=error

    def proc(self):
        Ntr=np.array_split(self.train, self.k)
        Nte=np.array_split(self.test, self.k)
        
        l=[None]*len(self.train[0])
        l=np.array(l)
        lt=[None]*len(self.test[0])
        lt=np.array(lt)
        

        alltrain=[]
        alltest=[]
        testpred=[]
        testout=[]
        for i in range(self.k):
            for j in range(self.k):
                if j ==i:
                    testpred.append(Ntr[j])
                    testout.append(Nte[j])
                    pass
                else:
                    l=np.row_stack((l,Ntr[j]))
                    lt=np.row_stack((lt,Nte[j]))
                    #print(Ntr[j])                    
                    #print(" ")
            #print("end")
            l=np.delete(l,(0),axis=0)
            lt=np.delete(lt,(0),axis=0)
            alltrain.append(l)
            alltest.append(lt)
            l=[None]*len(self.train[0])
            l=np.array(l)
            lt=[None]*len(self.test[0])
            lt=np.array(lt)
        

            
            
       # print(alltrain)
       # print(alltest)
       # print(testpred)
       # print(testout)
        self.alltrain=alltrain
        self.alltest=alltest
        self.testpredict=testpred
        self.testout=testout
        

    
    def eval(self):
        cost=[]
        costtest=[]
        
        
        #self.model.fit(self.train,self.test)
        #plt.plot(idx,self.model.costhistoryRMSE)
        if self.error=="RMSE":
            
            for i in range(self.k):
                self.model.kfit(self.alltrain[i],self.alltest[i],self.testpredict[i],self.testout[i],"RMSE")                
                cost.append(self.model.costhistoryRMSE)
                costtest.append(self.model.costhistoryRMSEtest)
        elif self.error=="MAE":
            for i in range(self.k):
                self.model.kfit(self.alltrain[i],self.alltest[i],self.testpredict[i],self.testout[i],"MAE")
                
                cost.append(self.model.costhistoryMAE)
                costtest.append(self.model.costhistoryMAEtest)

        min1=min([costtest[0][-1],costtest[1][-1],costtest[2][-1]])
        if min1==costtest[0][-1]:
            plt.plot(cost[0])
            plt.plot(costtest[0])
            plt.ylabel('Loss')
            plt.show()
            return self.alltrain[0],self.alltest[0],self.testpredict[0],self.alltest[0]
        elif min1==costtest[1][-1]:
            plt.plot(cost[1])
            plt.ylabel('Loss')
            plt.plot(costtest[1])
            plt.ylabel('Loss')            
            plt.show()
            return self.alltrain[1],self.alltest[1],self.testpredict[1],self.alltest[1]
        else:
            plt.plot(cost[2])
            plt.plot(costtest[2])
            plt.ylabel('Loss')
            plt.show()
            return self.alltrain[1],self.alltest[1],self.testpredict[2],self.alltest[2]
                 
                
            
        
    
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




'''x=MyPreProcessor()
X,y=x.pre_process(1)
#x=MyLogisticRegression()
#x.fit(X,y)
x=MyLinearRegression()
Y=Kfold(3,x,X,y,"RMSE")
Y.proc()                                #for Kfold plots, change the error to get rmse plot
Y.eval()  '''    
    
    
    




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





'''x=MyPreProcessor()              
X,y=x.pre_process(2)
train_x, validate_x, test_x = np.split(X, [int(.7 * len(X)), int(.8 * len(X))])      #train val test split for dataset 2
train_y, validate_y, test_y = np.split(y, [int(.7 * len(y)), int(.8 * len(y))])

x=MyLogisticRegression()

x.threefit(train_x,train_y,validate_x,validate_y,test_x,test_y,"BGD")            #Performing fit and this will plor graph of loss for both train and val set
#Default learning rate=1, epochs=10000

ans=x.predict(test_x)                            

print(np.count_nonzero(ans==test_y)/len(ans)*100) '''      #accuracy 


