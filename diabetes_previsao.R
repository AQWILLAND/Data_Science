#Previs�o para diagn�stico de diabete

### Carregando os dados e pacotes - Utilizando os dados contendo na pr�pria biblioteca do R

#Carregando pacotes
install.packages('mlbench')
library(mlbench)
library(dplyr)
library(caret)
library(ROCR) 
library(e1071)
install.packages("multiROC")
library(multiROC)

#Carregando os dados - Dados s�o diagn�sticos de diabetes atr�ves de 8 vari�veis n�mericas
data("PimaIndiansDiabetes")
diabete <- PimaIndiansDiabetes

#An�lise explorat�ria dos dados
dim(diabete)
glimpse(diabete)
summary(diabete)

#Verificando se h� valores missing
table(is.na(diabete))

#Transformando a var�avel target em factor para ajustar melhor ao modelo
diabete <- diabete %>%
  mutate(diabetes = ifelse(diabetes == "pos",1,0))

diabete$diabetes <- (diabete$diabetes)
  
glimpse(diabete)

#An�lise gr�fica para verificar as distribui��es

#Coletando os nomes das vari�veis que quero verificar
colnames(diabete)
nomes <- c(colnames(diabete[1:8]))
nomes

#Gr�fico de Histograma
par(mfrow=c(2,2))
for (i in nomes){
  
  print(i)
  hist(diabete[[i]],main = i)
  
}

#Gr�fico de Boxplot
par(mfrow=c(1,2))
for (i in nomes){
  
  print(i)
  boxplot(diabete[[i]],main = i)
  
}

#Atrav�s dos graficos de bloxplot, foi verificado a presen�a de outliers em quase todas as vari�veis
#Ser� feita a limpeza desses outliers

###Fazendo a limpeza dos dados - Remo��o dos outliers

f = 1
while (f < 2) {
  
  
  
  for (i in nomes){
    
    outlier = boxplot(diabete[[i]], plot = FALSE)$out
    
    diabete_1 <- diabete[-which(diabete[[i]] %in% outlier),]
  
    f = f+1
    
    print(i)
    print(dim(diabete_1))
    
  break
    
    next
  }
  
  for (f in nomes) {
    
    if(f==i){
      next
    } 
    
    else{
    
    outlier <- boxplot(diabete_1[[f]], plot = FALSE)$out
    
    diabete_1 <- diabete_1[-which(diabete_1[[f]] %in% outlier),]
    
    print(f)
    print(dim(diabete_1))
    
    }
  }
}


#Gr�fico de Boxplot ap�s limpeza
par(mfrow=c(1,2))
for (i in nomes){
  
  print(i)
  boxplot(diabete_1[[i]],main = i)
}

#Verificando o balanceamento da vari�vel target
table(diabete_1$diabetes)

#Analisado que est� bem desbalancedado a vari�vel, ser� feito um tratamento para baleancear para n�o haver 
#problemas de overfitting do modelo
install.packages('smotefamily')
library(smotefamily)

smote_1 <- SMOTE(diabete_1,diabete_1$diabetes, K = 5, dup_size = 0)
diabete_smote <- smote_1$data
table(diabete_smote$diabetes)
diabete_smote <- diabete_smote[,1:9]

#Transformando em factor a vari�vel target

diabete_smote$diabetes <- as.factor(diabete_smote$diabetes)

glimpse(diabete_smote)

#Normalizar as vari�veis num�ricas 

to.factors <- function(df,variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]])
  }
  return(df)
}

diabete_normalize <- to.factors(diabete_smote,nomes)


#Separar dados de treino e teste
indice_separacao <- sample(x = nrow(diabete_normalize),
                           size = 0.8 * nrow(diabete_normalize),
                           replace = FALSE)

#Separando os dados de treino e teste conforme �ndice
dados_treino <- diabete_normalize[indice_separacao,]
dados_teste <- diabete_normalize[-indice_separacao,]

dim(dados_teste)

dim(dados_treino)

###Fazendo o treinamento do primeiro modelo

# Construindo o modelo de regress�o log�stica
formula.init <- "diabetes ~ ."
formula.init <- as.formula(formula.init)
modelo_v1 <- glm(formula = formula.init, data = dados_treino, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_v1)

# Fazendo previs�es e analisando o resultado
previsoes <- predict(modelo_v1, dados_teste, type = "response")
previsoes <- round(previsoes)
View(previsoes)

# Confusion Matrix
confusionMatrix(table(data = previsoes, reference = dados_teste$diabetes))



###Segunda vers�o do modelo = utilizando apenas o SVM ###
modelo_v2 <- svm(diabetes ~.,
                 data = dados_treino,
                 na.action = na.omit,
                 scale = TRUE)
print(modelo_v2)

#Fazendo previs�es
previsoes_v2 <- predict(modelo_v2,newdata = dados_teste)

#Verificando m�tricas para verificar o desempenho do algoritmo
confusionMatrix(table(previsoes_v2,dados_teste$diabetes))


###Terceira vers�o do modelo = Utilizando o m�todo GRID SEARCH para encontra os melhor par�metros###
modelo_v3 <- tune(svm,
                  diabetes ~.,
                  data = dados_treino,
                  kernel = "linear",
                  ranges = list(cost = c(0.05,0.1,0.5,1,2)))
modelo_v3$best.parameters
modelo_v3$best.model
modelo_v3_bestparameter <- modelo_v3$best.model
modelo_v3_bestparameter
summary(modelo_v3_bestparameter)

#Fazendo previs�es
previsoes_v3 <- predict(modelo_v3_bestparameter,newdata = dados_teste)

# Confusion Matrix
confusionMatrix(table(previsoes_v3,dados_teste$diabetes))


###Escolha do modelo_v2 devido o mesmo ter uma accuracy maior entre os outros###
