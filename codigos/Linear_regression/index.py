#Objetivo: Aplicar diversos algortimos no dataset iris. Assim conhecer e interpretar os algortmos.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs

iris = pd.read_csv('~/Data_Science_e_Machine_Learning/dados/iris.csv')

especies =  iris.loc[:,['species']].groupby('species').count()
especies['NUM_CLASS'] = range(len(especies))

iris = iris.join(especies,on='species')
correlacao = iris.corr()

sbs.heatmap(correlacao)
sbs.pairplot(iris)
plt.show()
# separação dos dados
from sklearn.model_selection import train_test_split
x = iris.loc[:,['sepal_length','sepal_width','petal_length','petal_width']]  
y = iris.loc[:,['NUM_CLASS']]

# normalização de dados
x = x.astype(float)
y = y.astype(float)

x_treino,x_teste,y_treino,y_teste = train_test_split(x,y,test_size=0.3)
# instanciação da classe de regressão
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

#geração do modelo/treino

lm.fit(x_treino,y_treino)

predict = lm.predict(x_teste)
Classificar =  np.array([np.around(x) for x in predict])
sbs.distplot((y_teste - predict))
plt.show()

from sklearn.metrics import mean_squared_error

print('Predict Mean Squared Error:')
print(mean_squared_error(y_teste,predict))
print('\nintercept_: ')
print(lm.intercept_)
print('\ncoef_: ')
print(lm.coef_)
print('\n---------------Resultado final----------------------\n')
print('Y\tNao Classificado ->\t\tClassifido\n')
for i in range(0,45):    
    print('{}\t{} -------> {}'.format(y_teste.iloc[i].values,predict[i],Classificar[i]))

