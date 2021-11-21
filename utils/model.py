import numpy as np



class Perceptron: # Class
#Now we will define all the methods that we are going to use
    def __init__(self,eta,epochs):
        #Intialising preceptron
        self.weights = np.random.randn(3)* 1e-4 #Randomly intialised small weights and multyplying with 10^-4 
        print(f'initial weights before tarining: {self.weights}')
        
        self.eta = eta #Learning rate
        self.epochs = epochs #number of epochs
        
    
    
    
    def activationFunction(self,inputs,weights):
        z = np.dot(inputs,weights)  #z = W* X
        return np.where(z>0,1,0) # CONDITION, IF TRUE, ELSE
            
        
        
        
    def fit(self,X,y):
        self.X = X 
        self.y = y 
        
        X_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]  #Adding bias values to X matrix, concatation
        
        print(f'X with bias: \n{X_with_bias}')
        
        #Iterations: Training
        for epoch in range(self.epochs):
            print('--'*10)
            print(f"for epoch: {epoch}")
            print('--'*10)
            
            y_hat = self.activationFunction(X_with_bias,self.weights)   #Forward Propagation
            print(f'predicted values after forward pass: \n{y_hat}')
            
            self.error = self.y - y_hat
            print(f'error: \n{self.error}')
            
            self.weights = self.weights + self.eta*np.dot(X_with_bias.T,self.error)  #Backward Propagation
            print(f"updated weights after epoch:\n{epoch}/{self.epochs} : \n{self.weights}")
            print('#####'*10)
            
        
    
    
    
    
    def predict(self,X):
        X_with_bias = np.c_[X,-np.ones((len(X),1))] #Here it will be X, not self.X, because if
        #we used self.X that will mean
        #that the same value that is taken as input and has been used as self.x till now will
        #be used for predictions also.
        #X not self.X, so that the values for X for which wwe want predictions that can be taken 
        return self.activationFunction(X_with_bias,self.weights)
    
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f'Total loss: {total_loss}')
        return total_loss