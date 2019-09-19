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
plt.title('Correlação de variaveis')
plt.show()
sbs.pairplot(iris)
plt.title("Pairplot")
plt.show()
# separação dos dados
from sklearn.model_selection import train_test_split
x = iris.loc[:,['sepal_length','sepal_width','petal_length','petal_width']]  
y = iris.loc[:,['NUM_CLASS']]

# normalização de dados
x = x.astype(float)
y = y.astype(float)

x_treino,x_teste,y_treino,y_teste = train_test_split(x,y,test_size=0.3)

#----------------------------------------------
# instanciação da classe de regressão
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

#treino

lm.fit(x_treino,y_treino)
#predição
Predict = lm.predict(x_teste)
#----------------------------------------------
#adicionando uma função de classificação(decisão) 0,1,2
Predict_Classificado =  np.array([np.around(np.abs(x)) for x in Predict])
#----------------------------------------------

print('\nintercept_: ')
print(lm.intercept_)
print('\ncoef_: ')
print(lm.coef_)
print('\n---------------Resultado final----------------------\n')

print("\n-------------Legenda--------------\n")
print(especies)
print("\n----------------------------------\n")
print('Y\tPredict ->\t\tPredict\tClassificado\n')
for i in range(0,45):    
    print('{} -------> \t{}\t{}'.format(y_teste.iloc[i].values,Predict[i],Predict_Classificado[i]))

from sklearn.metrics import confusion_matrix
mt_confucao = confusion_matrix(y_true=y_teste,y_pred=Predict_Classificado)
plt.matshow(mt_confucao)
plt.title('Matriz de confução')
plt.ylabel('Classificações corretas')
plt.xlabel('Classificações obtidas')
plt.show()
x_range = [x for x in range(0,45)]
plt.scatter(y=y_teste.values,x=x_range)
plt.plot(Predict)
plt.title('Linear regression: Y ')
plt.show()
plt.scatter(y=y_teste.values,x=x_range)
plt.plot(Predict_Classificado)
plt.title("Classificado")
plt.show()