import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt 

def outFile():
    newFile = pd.read_csv("population.csv")
    worldPopulationData = newFile[(newFile['Country Name'] == 'World')]
    size = worldPopulationData.shape
    
    '''
    year = list(range(20))
    linear_reg_model = LinearRegression()
    x_Val = worldPopulationData['Year']
    y_Val = worldPopulationData['Value']
    linear_reg_model.fit(np.array((x_Val)).reshape(-1,1),np.array((y_Val)).reshape(-1,1))
    print(linear_reg_model.score(np.array((x_Val)).reshape(-1,1),np.array((y_Val)).reshape(-1,1)))
    
    plt.plot(np.array(x_Val),np.array(y_Val),'o')
    plt.show()
    '''
    
    

    _limiter = 1990
    newDF = pd.DataFrame(columns = ['Country Name', 'Country Code','Year', 'Value'])
    print(newFile.shape)
    
    for i in range(0,14885):
                      
        if(int(newFile.iloc[i,2])==_limiter):
            print('Year Now',newFile.iloc[i,2])
            rowAtm = newFile.iloc[[i]]            
            newDF = pd.concat([newDF, rowAtm])                
            i =+ 1
            _limiter = _limiter + 1
            print('limiter now:',_limiter )
            
            if(_limiter == 2017):
                _limiter = 1990
    '''            
    #newDF.to_csv('population_to_sixtin.csv',index = False, encoding = 'utf-8')
outFile()
