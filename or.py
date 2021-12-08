from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd 
import numpy as np
import logging
import os

logging_str = "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"  #format of logging string 
#logging.basicConfig(level=logging.INFO, format=logging_str)  #configuring log 
#Logging directory for storing logging information
log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)
logging.basicConfig(filename=os.path.join(log_dir,'running_logs.log'),level=logging.INFO,format=logging_str,filemode="a") #Filemode = a means appending all logs


def main(data,eta,epochs,filename,plotfilename):
    

    df = pd.DataFrame(data)
    logging.info(f'This is actual datframe{df}')
    X,y = prepare_data(df)

    model = Perceptron(eta ,epochs) #MAking an object model from class perceptron 
    model.fit(X,y) #Fit method of Perceptron class
    _ = model.total_loss()

    save_model(model,filename)
    save_plot(df,plotfilename,model)


if __name__ =='__main__':   ## Entry point for this file
    OR = {
        'x1':[0,0,1,1],
        'x2': [0,1,0,1],
        'y': [0,1,1,1]

    }
    ETA = 0.3 #Between 0 and 1
    EPOCHS = 10
    
    try:
        logging.info('>>>>>>>starting traing here>>>>>>>>>>')
        main(data = OR,eta=ETA,epochs=EPOCHS,filename='or.model',plotfilename='or.png')     ### Start execution from here
        logging.info('<<<<<<training done successfully<<<<<<<<<\n')
    except Exception as e:
        logging.exception(e)  #Logging exception
        raise(e)