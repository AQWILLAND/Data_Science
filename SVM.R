
####Prevendo gastos com cartão de créditos em 3 categorias - Supervisionados

####Será utilizado o modelo SVM para realização dessa previsões


#Instalando pacotes
install.packages("gains")
install.packages("pROC")
install.packages("ROSE")
install.packages("mice")

###Carregando pacotes###
library(DMwR)
library(dbplyr)
library(caret)
library(gains)
library(pROC)
library(ROSE)
library(mice)
library(ROCR)
library(e1071)

###Carregando dados###
dataset_1 <- read.csv("cartoes_clientes.csv")
View(dataset_1)

###Análise Exploratório dos Dados###
str(dataset_1)
summary(dataset_1$card2benefit)

###Removendo coluna ID###
dataset_2 <- dataset_1[-1]
dataset_2
str(dataset_2)
View(dataset_2)



###Pré-processamento dos Dados###

###Função para Fatorização de variáveis categóricas###
to.factors <- function(df,variables){
  for (variable in variables){
    df[[variable]] <- as.factor(paste(df[[variable]]))
  }
  return(df)
}

###Definindo as variáveis categóricas para realizar a fatorização###
categoric_variables <- c("townsize","jobcat","retire","hometype","addresscat",
                         "cartype","carvalue","carbought","card2","card2type",
                         "card2benefit","bfast","internet","Customer_cat")

dataset_2 <- to.factors(df = dataset_2, variables = categoric_variables)
str(dataset_2)
class(dataset_2$townsize)
class(dataset_2$jobcat)
table(sapply(dataset_2, is.factor))

###Checando valores missing###
sum(is.na(dados_teste))
sapply(dataset_2,function(x)sum(is.na(x)))

###Trabalhando valores missing através da técnica de imputação###
#Descobrindo os números das colunas fatores para excluí-las da imputação
fac_col <- as.integer(0)
facnames <- names(Filter(is.factor,dataset_2))
k = 1


for (i in facnames){
  while (k <= 15){
    grep(i,colnames(dataset_2))
    fac_col[k] <- grep(i,colnames(dataset_2))
    k = k+1
    break
  }
}


fac_col

#Definindo a regra de imputação
regra_imputacao <- mice((dataset_2[,-c(fac_col)]),
                        m = 1,
                        maxit = 50,
                        meth = 'pmm',
                        seed = 500)

total_data <- complete(regra_imputacao,1)

dataset_final <- cbind(total_data,dataset_2[,c(fac_col)])



####Dividindo entre dados de treino e teste###

#Montando o índice para divisão dos dados
indice_separacao <- sample(x = nrow(dataset_final),
                           size = 0.8 * nrow(dataset_final),
                           replace = FALSE)

#Separando os dados de treino e teste conforme índice
dados_treino <- dataset_final[indice_separacao,]
dados_teste <- dataset_final[-indice_separacao,]

###Verificando se a variável alvo está balanceada###
as.data.frame(table(dados_treino$Customer_cat))
prop.table(table(dados_treino$Customer_cat))*100

###Realizando o balanceamento da variável target utilizando a técnica SMOTE###
dados_treino_balanceados <- SMOTE(Customer_cat ~ .,dados_treino, perc.over = 3000, perc.under = 200)
prop.table(table(dados_treino_balanceados$Customer_cat))

#Transformando a variável target em número/fator
dados_treino_balanceados$Customer_cat <- as.numeric(as.factor(dados_treino_balanceados$Customer_cat))
dados_treino_balanceados$Customer_cat <- as.factor(dados_treino_balanceados$Customer_cat)

dados_teste$Customer_cat <- as.numeric(as.factor(dados_teste$Customer_cat))
dados_teste$Customer_cat <- as.numeric(dados_teste$Customer_cat)

###Modelagem preditiva utilizando o modelo SVM###

###Primeira versão do modelo = utilizando apenas o SVM ###
modelo_v1 <- svm(Customer_cat ~.,
                 data = dados_treino_balanceados,
                 na.action = na.omit,
                 scale = TRUE)
print(modelo_v1)

#Fazendo previsões
previsoes_v1 <- predict(modelo_v1,newdata = dados_teste)

#Verificando métricas para verificar o desempenho do algoritmo
caret::confusionMatrix(table(previsoes_v1,dados_teste$Customer_cat))

install.packages("multiROC")
library(multiROC)

curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(previsoes_v1))
curva_roc$auc

###Segunda versão do modelo = Utilizando o método GRID SEARCH para encontra os melhor parâmetros###
modelo_v2 <- tune(svm,
                  Customer_cat ~.,
                  data = dados_treino_balanceados,
                  kernel = "linear",
                  ranges = list(cost = c(0.05,0.1,0.5,1,2)))
modelo_v2$best.parameters
modelo_v2$best.model
modelo_v2_bestparameter <- modelo_v2$best.model
modelo_v2_bestparameter
summary(modelo_v2_bestparameter)

#Fazendo previsões
previsoes_v2 <- predict(modelo_v2_bestparameter,newdata = dados_teste)

#Verificando métricas para verificar o desempenho do algoritmo
caret::confusionMatrix(table(previsoes_v2,dados_teste$Customer_cat))

curva_roc_2 <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(previsoes_v2))
curva_roc_2$auc

###Terceira versão do modelo = Utilizando o método GRID SEARCH para encontra os melhor parâmetros###
modelo_v3 <- tune(svm,
                  Customer_cat ~.,
                  data = dados_treino_balanceados,
                  kernel = "radial",
                  ranges = list(cost = c(0.05,0.1,0.5,1,2),
                                gamma = c(0.001,0.01,0.1,0.5,1,2)))
modelo_v3_bestparameter <- modelo_v3$best.model

#Fazendo previsões
previsoes_v3 <- predict(modelo_v3_bestparameter,newdata = dados_teste)

#Verificando métricas para verificar o desempenho do algoritmo
caret::confusionMatrix(table(previsoes_v3,dados_teste$Customer_cat))

curva_roc_3 <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(previsoes_v3))
curva_roc_3$auc


###Quarta versão do modelo = Utilizando o método GRID SEARCH para encontra os melhor parâmetros###
modelo_v4 <- tune(svm,
                  Customer_cat ~.,
                  data = dados_treino_balanceados,
                  kernel = "polynomial",
                  ranges = list(cost = c(0.5,1,2),degree = c(2,3,4)))
modelo_v4_bestparameter <- modelo_v4$best.model

#Fazendo previsões
previsoes_v4 <- predict(modelo_v4_bestparameter, newdata = dados_teste)

#Verificando métricas para verificar o desempenho do algoritmo
caret::confusionMatrix(table(previsoes_v4,dados_teste$Customer_cat))

curva_roc_4 <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(previsoes_v4))
curva_roc_4$auc


####Realizado 4 modelos de algorimo e o com rsultado mais satisfatório foi o segundo -> 
####modelo com a ACURÁCIA e métrica ROC maior. Segue abaixo modelo escolhido...

###Segunda versão do modelo = Utilizando o método GRID SEARCH para encontra os melhor parâmetros###
modelo_v2 <- tune(svm,
                  Customer_cat ~.,
                  data = dados_treino_balanceados,
                  kernel = "linear",
                  ranges = list(cost = c(0.05,0.1,0.5,1,2)))
modelo_v2$best.parameters
modelo_v2$best.model
modelo_v2_bestparameter <- modelo_v2$best.model
modelo_v2_bestparameter
summary(modelo_v2_bestparameter)

#Fazendo previsões
previsoes_v2 <- predict(modelo_v2_bestparameter,newdata = dados_teste)

#Verificando métricas para verificar o desempenho do algoritmo
caret::confusionMatrix(table(previsoes_v2,dados_teste$Customer_cat))

curva_roc_2 <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(previsoes_v2))
curva_roc_2$auc
