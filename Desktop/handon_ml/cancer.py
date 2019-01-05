import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import scale 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from matplotlib import pyplot as plt


def cancerAnalysis():
    #importing all the data 
    file1 = pd.read_csv('FiveYearCancerSurvivalRateinUSA.csv').dropna(how='all')    
    file2 = pd.read_csv('cancerWholeWorld.csv').dropna(how='all')
    populationList = pd.read_csv('population_to_sixtin.csv').drop('Country Code',axis = 1)
    
    countryListAll = file2.iloc[:,0]
    survivalRateFemale = file1.iloc[:11,0] 
    yearFemale =file1.iloc[:11,1]
    survivalRateMale=file1.iloc[12:22,0]    
    yearMale = file1.iloc[12:22,1]
    survivalRateTotal=file1.iloc[22:33,0]
    yearTotal = file1.iloc[22:33,1]

    yearsForPrediction = [list(range(1990,2020))]

    countryList =[(countryListAll[0])]
    cancerList = list(file2.columns.values)[3:]
    i=0
    j=1 
    caseForLungCancer = file2[['Tracheal, bronchus, & lung cancer (%)','Entity','Year']]
    
    
    #To make a total list of country
    while( j != len(countryListAll)):   
        if(countryList[i]!=countryListAll[j]):           
            countryList.append(countryListAll[j])
            i+=1
            j+=1                 
        else:
            j+=1
    #for i in countryListAll:
        
        
    
    CancerForMongolia = file2[(file2['Entity']=='Mongolia')]
    CancerForAustralia = file2[(file2['Entity']=='Australia')] 
    CancerForNepal = file2[(file2['Entity']=='Nepal')]


    label_encoder = preprocessing.LabelEncoder
    
    #Prediction for liverCancer in Mongolia
    lin_reg_model = LinearRegression()
    logistic_reg_model = LogisticRegression()
    tree_reg_model = DecisionTreeRegressor(max_depth=3)
    X_value_for_Mongolia = np.array(CancerForMongolia['Year'].apply(pd.to_numeric, errors = 'coerce')).reshape(-1,1)
    #X_value_for_logReg = np.array(list(zip(CancerForMongolia['Year'].apply(pd.to_numeric, errors = 'coerce'),populationMongolia['Value'].apply(pd.to_numeric, errors = 'coerce'))))
    #print(X_value_for_logReg)
    #encoded_X = label_encoder.fit_transform(X_value_for_logReg)
    y_value_for_Mongolia = np.array(CancerForMongolia['Liver cancer (%)'].apply(pd.to_numeric, errors = 'coerce')).reshape(-1,1)
    X_value_for_Aussie = np.array(CancerForAustralia['Year'].apply(pd.to_numeric, errors = 'coerce')).reshape(-1,1)
    y_value_for_Aussie = np.array(CancerForAustralia['Non-melanoma skin cancer (%)'].apply(pd.to_numeric,errors = 'coerce')).reshape(-1,1)
    
   
    
    #For Mongolia
    lin_reg_model.fit(X_value_for_Mongolia,y_value_for_Mongolia)
    #logistic_reg_model.fit(encoded_X,y_value_for_Mongolia.ravel(order ='C'))  
    tree_reg_model.fit(X_value_for_Mongolia,y_value_for_Mongolia.ravel(order= 'C'))#order C--> means row major transformation  
    
    
    predicted_Y_val_linReg = lin_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))
    predicted_Y_val_Dtree =tree_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))
    
    
    #print('logistic regression fitting score value:',logistic_reg_model.score(X_value_for_logReg,y_value_for_Mongolia.ravel(prder ='C')))
    print('linear Regression fitting score: ',lin_reg_model.score(X_value_for_Mongolia,y_value_for_Mongolia))
    print('Decision Tree Regression fitting score: ',tree_reg_model.score(X_value_for_Mongolia,y_value_for_Mongolia))
    #print()
    
    
          

    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(10,8))
        plt.plot(np.array(CancerForMongolia['Year']),np.array(CancerForMongolia['Liver cancer (%)']),'^g',label = "Liver Cancer ")
        #plt.plot(np.array(yearsForPrediction).reshape(-1,1).flatten(),predicted_Y_val_linReg.flatten(),'o', label = "Liver Cancer Predicted LinREG")        
        #plt.plot(np.array(yearsForPrediction).reshape(-1,1).flatten(),predicted_Y_val_Dtree.flatten(),'x', label = "Liver Cancer Predicted Decision Tree")
    plt.xlabel('Graph for cancer in Mongolia from 1990 to 2019 ')
    plt.legend(loc = 'lower right')
    plt.show()     

    lin_reg_model.fit(X_value_for_Aussie,y_value_for_Aussie)
    tree_reg_model.fit(X_value_for_Aussie,y_value_for_Aussie.ravel(order = 'C'))
    
    predicted_Y_val_linReg = lin_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))
    predicted_Y_val_Dtree =tree_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))
    
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(10,8))
        plt.plot(np.array(CancerForAustralia['Year']),np.array(CancerForAustralia['Non-melanoma skin cancer (%)']),'^g',label = "Skin Cancer")
        #plt.plot(np.array(yearsForPrediction).reshape(-1,1).flatten(),predicted_Y_val_linReg.flatten(),'o', label = "Skin Cancer Predicted LinREG")        
        #plt.plot(np.array(yearsForPrediction).reshape(-1,1).flatten(),predicted_Y_val_Dtree.flatten(),'x', label = "Skin Cancer Predicted LinSVR")
    plt.xlabel('Graph for cancer in Australia from 1990 to 2019 ')
    plt.legend(loc = 'lower right')
    plt.show()
    
    predicted_Y_val_linReg = lin_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))
    predicted_Y_val_Dtree =tree_reg_model.predict(np.array(yearsForPrediction).reshape(-1,1))      
    
    caseFor1990 = file2[(file2['Year']==1990)]
    caseFor1990_sk = caseFor1990[['Non-melanoma skin cancer (%)','Entity']]
    caseFor1990_lk = caseFor1990[['Liver cancer (%)','Entity']]
    caseFor1990_lk = caseFor1990_lk.set_index('Entity')
    caseFor1990_sk = caseFor1990_sk.set_index('Entity')   
    caseFor2016 = file2[(file2['Year']==2016)]
    caseFor2016_sk = caseFor2016[['Non-melanoma skin cancer (%)','Entity']]
    caseFor2016_lk = caseFor2016[['Liver cancer (%)','Entity']]
    caseFor2016_lk = caseFor2016_lk.set_index('Entity')
    caseFor2016_sk = caseFor2016_sk.set_index('Entity')
    
    
    plt.figure(figsize=(20,20))
    caseFor1990_sk.plot.pie(y = 'Non-melanoma skin cancer (%)',figsize = [15,15], legend = False )
    caseFor1990_lk.plot.pie(y = 'Liver cancer (%)',figsize = [15,15], legend = False )
    caseFor2016_sk.plot.pie(y = 'Non-melanoma skin cancer (%)',figsize = [15,15], legend = False )
    caseFor2016_lk.plot.pie(y = 'Liver cancer (%)',figsize = [15,15], legend = False )
    plt.show()
    
    
    print('Total number of people in australia suffering from skin cancer(1990): ',(populationList.iloc[297,2]*caseFor1990_sk.iloc[11,0]))
    print('Total number of people in australia suffering from skin cancer(2016): ',(populationList.iloc[323,2]*caseFor2016_sk.iloc[11,0]))
    print('Total number of people in mongolia suffering from liver cancer(1990): ',(populationList.iloc[3510,2]*caseFor1990_lk.iloc[133,0]))
    print('Total number of people in mongolia suffering from liver cancer(2016): ',(populationList.iloc[3536,2]*caseFor2016_lk.iloc[133,0]))
         
        
    
    
    
    for i in range(len(cancerList)):
        plt.figure(figsize=(35,20))
        plt.bar(np.array(caseFor1990['Entity']),np.array(caseFor1990[cancerList[i]]),align='edge', width = 0.8, alpha = 0.8)
        plt.xticks(np.array(caseFor1990['Entity']),caseFor1990['Entity'], rotation =90, fontsize = 12, weight = 'bold',linespacing =0.89)
        plt.ylabel(cancerList[i])
        plt.xlabel('Country')
        plt.title('1990')        
        filename1990 = "1990 %s.png"%(cancerList[i])
        #print(filename1990)
        plt.savefig((filename1990),dpi=100) 
        plt.show()
        plt.figure(figsize=(35,20))
        plt.bar(np.array(caseFor2016['Entity']),np.array(caseFor2016[cancerList[i]]),align='edge', width = 0.8, alpha = 0.8)
        plt.xticks(np.array(caseFor2016['Entity']),caseFor2016['Entity'], rotation =90,fontsize = 12, weight = 'bold', linespacing = 0.89)
        plt.ylabel(cancerList[i])
        plt.xlabel('Country')
        plt.title('2016')                
        filename2016 ="2016 %s.png"%(cancerList[i])
        #print(filename2016)
        plt.savefig((filename2016),dpi =100)
        plt.show()
    with plt.style.context('fivethirtyeight'): 
        plt.plot(np.array(yearFemale), np.array(survivalRateFemale), 'r--',label = "female")
        plt.plot(np.array(yearMale), np.array(survivalRateMale), 'bs',label = "male")
        plt.plot(np.array(yearTotal),np.array(survivalRateTotal),'g^',label = "total")
    plt.ylabel('Survival Rate')
    plt.xlabel('Year')
    plt.legend(loc='lower right')
    plt.show()
    
    #print(countryList)

    
def coefficientCorelation(x_data, y_data):
    '''
    this function return the coefficent corellation of two data sets with the same length of data point 
    '''
    if(len(x_data) == len(y_data)):
        n = len(x_data) # total number of data points 
        sum_x= 0
        sum_y= 0
        sum_xy = 0 
        sum_squaredX = 0
        sum_squaredY = 0
        for i in range(len(x_data)):
            sum_x = x_data[i] + sum_x
            sum_y = y_data[i] + sum_y
            sum_xy = (x_data[i]*y_data[i]) + sum_xy
            sum_squaredX = (x_data[i]**2) + sum_squaredX
            sum_squaredY = (y_data[i]**2) + sum_squaredY            
            i += 1
        r = (n*sum_xy)-((sum_x)*(sum_y))/(math.sqrt(((n*sum_squaredX)-(sum_x**2))*((n*sum_squaredY)-(sum_y**2))))
        return r
    else:
        return "The data points are not enough."

    

#def populationIncrease():
    


    
from dis import dis
outfile = file.write('Assembly.txt','w')
print(dis(cancerAnalysis))    
    
    
#cancerAnalysis()   