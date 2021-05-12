# Carregando os pacotes
library(caret)
library(ROCR) 
library(e1071)
library(dplyr)

#Negócio - Prever a variável Custumer Churn do dataset - Utilzando Regressão logística

#Carregando os dados
dados_treino <- read.csv("projeto4_telecom_treino.csv", header = TRUE)
glimpse(dados_treino)

#Retirando primeira coluna
dados_treino <- dados_treino[,-1]
dados_treino

#Transformando variáveis string em números
#Variáveis que precicos mudar: area_code, international_plan, voice_mail_plan, churn

table(dados_treino$voice_mail_plan)

dados_treino <- dados_treino %>%
  mutate(area_code = 
           case_when(area_code == "area_code_408" ~ 1,
                     area_code == "area_code_415" ~ 2,
                     area_code == "area_code_510" ~ 3))

dados_treino <- dados_treino %>%
  mutate(international_plan = ifelse(international_plan == "yes",1,0))

dados_treino <- dados_treino %>%
  mutate(voice_mail_plan = ifelse(voice_mail_plan == "yes",1,0))

dados_treino <- dados_treino %>%
  mutate(churn = ifelse(churn == "yes",1,0)) 

glimpse(dados_treino)

#Criando funções para normalização e fatorização

# Transformando variáveis em fatores
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

#factor
features_factors <- c("area_code", "international_plan", "voice_mail_plan", "churn")
dados_treino_1 <- to.factors(dados_treino,features_factors)

#normalizar
features_numeric <- c("account_length","number_vmail_messages","total_day_minutes",
                      "total_day_calls","total_day_charge","total_eve_minutes",
                      "total_eve_calls","total_eve_charge","total_night_minutes",
                      "total_night_calls","total_night_charge","total_intl_minutes",
                      "total_intl_calls","total_intl_charge","number_customer_service_calls")
dados_treino_transformados <- scale.features(dados_treino_1,features_numeric)

glimpse(dados_treino_transformados)

#Aplicando os transformações nos dados de teste
dados_teste <- read.csv("projeto4_telecom_teste.csv",header = TRUE)
#Retirando primeira coluna
dados_teste <- dados_teste[,-1]

#Transformando variáveis string em números
dados_teste <- dados_teste %>%
  mutate(area_code = 
           case_when(area_code == "area_code_408" ~ 1,
                     area_code == "area_code_415" ~ 2,
                     area_code == "area_code_510" ~ 3))

dados_teste <- dados_teste %>%
  mutate(international_plan = ifelse(international_plan == "yes",1,0))

dados_teste <- dados_teste %>%
  mutate(voice_mail_plan = ifelse(voice_mail_plan == "yes",1,0))

dados_teste <- dados_teste %>%
  mutate(churn = ifelse(churn == "yes",1,0))

#Criando funções para normalização e fatorização
#factor
dados_teste1 <- to.factors(dados_teste,features_factors)

#normalizar
dados_teste_transformados <- scale.features(dados_teste1,features_numeric)

glimpse(dados_treino_transformados)
glimpse(dados_teste_transformados)

# Construindo o modelo de regressão logística
formula.init <- "churn ~ ."
formula.init <- as.formula(formula.init)
help(glm)
modelo_v1 <- glm(formula = formula.init, data = dados_treino_transformados, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_v1)

# Fazendo previsões e analisando o resultado
previsoes <- predict(modelo_v1, dados_teste_transformados, type = "response")
previsoes <- round(previsoes)
View(previsoes)

# Confusion Matrix
confusionMatrix(table(data = previsoes, reference = dados_teste_transformados$churn), positive = '1')

###uTILIZADO PORÉM SEM SUCESSO NO MODELO
#Balanceamento
#install.packages("DMwR")
#library(DMwR)
#smoted_data <- SMOTE(churn ~.,data = dados_treino_transformados, perc_min = 50, k = 5)


