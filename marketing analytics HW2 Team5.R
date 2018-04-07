# Team 5: Dawei Jia/ Siqi Zhi/ Yufei Jiang/ Cenli Han/ Jinchuan Yang
########################         Part1: Data Management          #####################################################################

#a) load data and save as data frame
db.base <- read.csv("/Users/jiadawei/Desktop/Market Analytics/Homework 2 - MKT436R Data.csv")

#b) define a empty data frame
consumerID <- unique(db.base$consumerID) #find unrepeated ID
length(customerID) # to determine the row number
Mymatrix <- matrix(data=NA,nrow =80245, ncol=6, dimnames = list(1:80245,c("consumerID","rocky1","rocky2","rocky3","rocky4","rocky5")) )
db <- data.frame(Mymatrix) #create an empty database
db$consumerID <- consumerID

#c) fill the data frame
library(reshape2)
db <- dcast(db.base,consumerID~rockyID,value.var = "rating") #fill the database
colnames(db) <- c("consumerID","rocky1","rocky2","rocky3","rocky4","rocky5")

########################         Part2: Data exploration and Sampling Bias          ##################################################################

#a) compare the correlation
cor(db[,2:6],use = "pairwise.complete.obs") #correlation betwen five rocky

#b) Find the mean rating of each movie
colMeans(db[,2:6],na.rm = T)

#c)Create a subset of your data frame that only contains consumers who rated Rocky 4.
db2 <- db[which(!is.na(db$rocky4)),]
colMeans(db2[,2:6],na.rm = T)

########################         Part3: Explanatory Models          ##################################################################

#load complete data base
completeDB <- read.csv("/Users/jiadawei/Desktop/Market Analytics/Homework 2 - completeDB.csv") 

#a)  generate different orders of interactions
firstInteractions <- model.matrix(~(- 1+rocky1+rocky2+rocky3+rocky4),completeDB)
secondInteractions <- model.matrix(~(- 1+rocky1+rocky2+rocky3+rocky4)^2,completeDB)
thirdInteractions <- model.matrix(~(- 1+rocky1+rocky2+rocky3+rocky4)^3,completeDB)
fourthInteractions <- model.matrix(~(- 1+rocky1+rocky2+rocky3+rocky4)^4,completeDB)

#b) Run and store a linear regression for each of the above sets of predictor variables.
firstlm <- lm(completeDB$rocky5~firstInteractions)
secondlm <- lm(completeDB$rocky5~secondInteractions)
thirdlm <- lm(completeDB$rocky5~thirdInteractions)
fourthlm <- lm(completeDB$rocky5~fourthInteractions)

#c) calculate AIC and BIC
AICvar <- c(AIC(firstlm),AIC(secondlm),AIC(thirdlm),AIC(fourthlm))
BICvar <- c(BIC(firstlm),BIC(secondlm),BIC(thirdlm),BIC(fourthlm))
Mergedtable <- rbind(AICvar,BICvar)
colnames(Mergedtable) <- c("firstlm","secondlm","thirdlm","fourthlm")
Mergedtable

#d) lasso regression
install.packages('glmnet')
library('glmnet')
lassoFit = glmnet(fourthInteractions, completeDB$rocky5,alpha=1)
plot(lassoFit)
predict(lassoFit,s = 0.5, type = 'coefficients')
predict(lassoFit,s = 0.05, type = 'coefficients')

#e) optimal penalty
lassoFit2 <- cv.glmnet(fourthInteractions, completeDB$rocky5,alpha=1)
plot(lassoFit2)

#f) ridge regression
ridgeFit <- cv.glmnet(fourthInteractions, completeDB$rocky5, alpha=0)
plot(ridgeFit)

#g) extract coefficient
predict(lassoFit2, s=lassoFit2$lambda.min, type = 'coefficients')
predict(ridgeFit, s=ridgeFit$lambda.min, type = 'coefficients')

predict(lassoFit,s = 0, type = 'coefficients')

########################         Part3: Predictive Models          ##################################################################

# a) using F-fold to create many training and validation data base
nFold <- 10
valnum <- floor(runif(nrow(completeDB))*nFold) +1
set.seed(1) # set seed to guarantee the same output
########################         b) Linear Regression          ######################################################################
modelperformance <- matrix(NA,nFold,32767) # to build a empty matrix to save output data. 32767 is total number of below models.

#--------------------  regressors combination of main effects and interactions ------------------------------------------------------

regressors <- c("rocky1","rocky2","rocky3","rocky4","I(rocky1*rocky2)","I(rocky1*rocky3)","I(rocky1*rocky4)",
                "I(rocky1*rocky4)","I(rocky2*rocky3)","I(rocky2*rocky4)","I(rocky3*rocky4)","I(rocky1*rocky2*rocky3)",
                "I(rocky1*rocky2*rocky4)","I(rocky2*rocky3*rocky4)","I(rocky1*rocky2*rocky3*rocky4)") # all regressors 
regMat <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                      c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                      c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE)) 
# use expand.gird to get all combinations of regressors above
regMat <- regMat[-(dim(regMat)[1]),] # delete the row of all FALSE otherwise there will be no matching error
meantable <- c() # build a empty vector to save output
allModelsList <- apply(regMat, 1, function(x) as.formula(paste(c("rocky5 ~ 1", regressors[x]),collapse=" + ")) )
# build lm() regression's bodies
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  allModelsResults <- lapply(allModelsList,function(x) lm(x, data=trainingdata))
  for (i in 1:32767) {meantable[i] <- mean((validationdata$rocky5-predict(allModelsResults[[i]],validationdata))^2)}
  modelperformance[fold,]<- meantable
}
# this for loop is to apply lm() to every model and calculate their MSE
which.min(colMeans(modelperformance)) #the minimum MSE is at row 1248
colMeans(modelperformance)[1248] #the minimum MSE is 0.9225057

#---------------------------  regressors combination of log(), square and some interactions ------------------------------------------------------

modelperformance2 <- matrix(NA,nFold,4095) # to build a empty matrix to save output data. 32767 is total number of below models.
regressors2 <- c("I(rocky1^2)","I(rocky2^2)","I(rocky3^2)","I(rocky4^2)","I(rocky1*rocky2*rocky3)","I(rocky1*rocky2*rocky4)","I(rocky2*rocky3*rocky4)",
                 "I(rocky1*rocky2*rocky3*rocky4)","log(rocky1)","log(rocky2)","log(rocky3)","log(rocky4)")
regMat2 <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                      c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                      c(TRUE,FALSE),c(TRUE,FALSE)) 
# use expand.gird to get all combinations of regressors above
regMat2 <- regMat2[-(dim(regMat2)[1]),] # delete the row of all FALSE otherwise there will be no matching error
meantable2 <- c() # build a empty vector to save output
allModelsList2 <- apply(regMat2, 1, function(x) as.formula(paste(c("rocky5 ~ 1", regressors2[x]),collapse=" + ")) )
# build lm() regression's bodies
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  allModelsResults2 <- lapply(allModelsList2,function(x) lm(x, data=trainingdata))
  for (i in 1:4095) {meantable2[i] <- mean((validationdata$rocky5-predict(allModelsResults2[[i]],validationdata))^2)}
  modelperformance2[fold,]<- meantable2
}
# this for loop is to apply lm() to every model and calculate their MSE
which.min(colMeans(modelperformance2)) #the minimum MSE is at row 1413
colMeans(modelperformance2)[1413] #the minimum MSE is 0.9103398

#---------------------------  regressors combination of log() and all interactions ------------------------------------------------------

modelperformance3 <- matrix(NA,nFold,16383) # to build a empty matrix to save output data. 32767 is total number of below models.
regressors3 <- c("log(rocky1)","log(rocky2)","log(rocky3)","log(rocky4)","I(rocky1*rocky2)","I(rocky1*rocky3)",
                 "I(rocky1*rocky4)","I(rocky2*rocky3)","I(rocky2*rocky4)","I(rocky3*rocky4)","I(rocky1*rocky2*rocky3)",
                 "I(rocky1*rocky2*rocky4)","I(rocky2*rocky3*rocky4)","I(rocky1*rocky2*rocky3*rocky4)")
regMat3 <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                       c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),
                       c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE),c(TRUE,FALSE)) 
# use expand.gird to get all combinations of regressors above
regMat3 <- regMat3[-(dim(regMat3)[1]),] # delete the row of all FALSE otherwise there will be no matching error
meantable3 <- c() # build a empty vector to save output
allModelsList3 <- apply(regMat3, 1, function(x) as.formula(paste(c("rocky5 ~ 1", regressors3[x]),collapse=" + ")) )
# build lm() regression's bodies
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  allModelsResults3 <- lapply(allModelsList3,function(x) lm(x, data=trainingdata))
  for (i in 1:16383) {meantable3[i] <- mean((validationdata$rocky5-predict(allModelsResults3[[i]],validationdata))^2)}
  modelperformance3[fold,]<- meantable3
}
# this for loop is to apply lm() to every model and calculate their MSE
which.min(colMeans(modelperformance3)) #the minimum MSE is at row 6424
colMeans(modelperformance3)[6424] #the minimum MSE is 0.9136714

# apply best linear regression model to entire dataset
bestlinear <- lm(rocky5 ~ 1 + I(rocky1^2) + I(rocky2^2) + I(rocky4^2) + I(rocky1 * rocky2 * rocky3) + 
                   I(rocky1 * rocky2 * rocky4) + I(rocky2 *rocky3 * rocky4) + log(rocky2) + log(rocky4), 
                 data=completeDB) # apply the best model to complete database

########################         b) MARS using earth package          #################################################################

install.packages("earth")
library(earth) #load earth package
#--------------------------              earth(..., trace=2, thres=0.1)    -------------------------------------------------------------

# earth(x, trace=2, thres=0.1)
set.seed(1)
modelperformance.earth1 <- matrix(NA,nFold,1) #build a empty matrix for output
for (fold in 1:nFold) {
    trainingdata <- subset(completeDB,valnum!=fold)
    validationdata <- subset(completeDB,valnum==fold)
    modelperformance.earth1[fold,] <- mean((validationdata$rocky5-predict(
      earth(rocky5~rocky1+rocky2+rocky3+rocky4, data = trainingdata,trace=2,thres=0.1),
      validationdata))^2)
}
#this for loop is to do earth by k-Fold cross validation
colMeans(modelperformance.earth1) # the MSE is 1.028126 #calculate the minimum MSE

# earth(x^2, trace=2, thres=0.1)
set.seed(1)
modelperformance.earth2 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth2[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1^2+rocky2^2+rocky3^2+rocky4^2, data = trainingdata,trace=2,thres=0.1),
    validationdata))^2)
}
colMeans(modelperformance.earth2) # 1.028126, the same as unsquare

# earth(log(x), trace=2, thres=0.1)
set.seed(1)
modelperformance.earth3 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth3[fold,] <- mean((validationdata$rocky5-exp(predict(
    earth(log(rocky5)~log(rocky1)+log(rocky2)+log(rocky3)+log(rocky4), data = trainingdata,trace=2,thres=0.1),
    validationdata)))^2)
}
colMeans(modelperformance.earth3) #the MSE is 1.072654

#--------------------------       basic fit  earth(...)    -------------------------------------------------------------

#earth(x, ...)
set.seed(1)
modelperformance.earth4 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth4[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1+rocky2+rocky3+rocky4, data = trainingdata),
    validationdata))^2)
}
colMeans(modelperformance.earth4) #the MSE is 0.9410418

#earth(x^2, ...)
set.seed(1)
modelperformance.earth5 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth5[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1^2+rocky2^2+rocky3^2+rocky4^2, data = trainingdata),
    validationdata))^2)
}
colMeans(modelperformance.earth5) #the MSE is 0.9410418

#earth(log(x), ...)
set.seed(1)
modelperformance.earth6 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth6[fold,] <- mean((validationdata$rocky5-exp(predict(
    earth(log(rocky5)~log(rocky1)+log(rocky2)+log(rocky3)+log(rocky4), data = trainingdata,trace=2,thres=0.1),
    validationdata)))^2)
}
colMeans(modelperformance.earth6) #the MSE is 1.072654 

#--------------------------       earth(...,degree=i)    ----------------------------------------------------------

#earth(x,degree=2,...)
set.seed(1)
modelperformance.earth7 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth7[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1+rocky2+rocky3+rocky4, data = trainingdata,degree = 2),
    validationdata))^2)
}
colMeans(modelperformance.earth7) #the MSE is 0.9093659

#earth(log(x),degree=3,...)
set.seed(1)
modelperformance.earth8 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth8[fold,] <- mean((validationdata$rocky5-exp(predict(
    earth(log(rocky5)~log(rocky1)+log(rocky2)+log(rocky3)+log(rocky4), data = trainingdata,degree = 3),
    validationdata)))^2)
}
colMeans(modelperformance.earth8) #the MSE is 0.9406891

#earth(log(x)+x^2,degree=3,...)
set.seed(1)
modelperformance.earth9 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth9[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~log(rocky1)+rocky2^2+rocky3^2+log(rocky4), data = trainingdata,degree = 3),
    validationdata))^2)
}
colMeans(modelperformance.earth9) #the MSE is 0.9108271

#earth(log(x)+x,degree=3,...)
set.seed(1)
modelperformance.earth10 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth10[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1+rocky2+rocky3+rocky4+log(rocky4), data = trainingdata,degree = 3),
    validationdata))^2)
}
colMeans(modelperformance.earth10) #the MSE is 0.9080266

#earth(log(x)+x,degree=3,...)
set.seed(1)
modelperformance.earth11 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth11[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1+rocky2+rocky3+rocky4+log(rocky3)+log(rocky4), data = trainingdata,degree = 3),
    validationdata))^2)
}
colMeans(modelperformance.earth11) #the MSE is 0.9047191

#earth(log(x)+x,degree=4,...)
set.seed(1)
modelperformance.earth12 <- matrix(NA,nFold,1)
for (fold in 1:nFold) {
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.earth12[fold,] <- mean((validationdata$rocky5-predict(
    earth(rocky5~rocky1+rocky2+rocky3+rocky4+log(rocky4), data = trainingdata,degree = 4),
    validationdata))^2)
}
colMeans(modelperformance.earth12) # the MSE is 0.9082063

# apply best MARS model to entire dataset
bestMARS <- earth(rocky5~rocky1+rocky2+rocky3+rocky4+log(rocky3)+log(rocky4), data = completeDB,degree = 3) # apply the best model to complete database

#######################################       KNN       ###################################################
install.packages('kknn')
library(kknn) # load knn package
set.seed(1)

#------------------------           linear distance         ------------------------------------------------

modelperformance.kknn1 <- matrix(NA,nFold,32767) #build a empty matrix for output
for (fold in 1:nFold) {
  for (j in 1:32767){
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.kknn1[fold,j] <- mean((validationdata$rocky5-(kknn(allModelsList[[j]], 
       trainingdata, validationdata, k=10, distance = 1))$fitted.values)^2)
  }}
#this for loop is to calculate all 32767 models at k=10, distance=1
colMeans(modelperformance.kknn1)[31740] #the minimun MSE is 1.004518
which.min(colMeans(modelperformance.kknn1)) #the minimun MSE is at row 31740

#--------------------------             square distance         -----------------------------------------

modelperformance.kknn2 <- matrix(NA,nFold,32767) #build a empty matrix for output
for (fold in 1:nFold) {
  for (j in 1:32767){
    trainingdata <- subset(completeDB,valnum!=fold)
    validationdata <- subset(completeDB,valnum==fold)
    modelperformance.kknn2[fold,j] <- mean((validationdata$rocky5-(kknn(allModelsList[[j]], 
               trainingdata, validationdata, k=5))$fitted.values)^2)
  }}
#this for loop is to calculate all 32767 models at k=5
colMeans(modelperformance.kknn2)[31744] #the minimun MSE is 1.081311
which.min(colMeans(modelperformance.kknn2)) #the minimun MSE is at row 31744

# apply best KKNN model to entire dataset
validationdata <- subset(completeDB,valnum==10)
bestkknn <- kknn(allModelsList[[31740]], completeDB, validationdata, k=10, distance = 1) # apply the best model to complete database

####################################### Neural networks ###############################################
install.packages("nnet")
library(nnet)
set.seed(1)

#----------------------------------        nnt(x, size=j)      -----------------------------------------------

modelperformance.nnt <- matrix(NA,nFold,50)
for (fold in 1:nFold) {
    for (j in 1:50){
  trainingdata <- subset(completeDB,valnum!=fold)
  validationdata <- subset(completeDB,valnum==fold)
  modelperformance.nnt[fold,j] <- mean((validationdata$rocky5-predict(
    nnet(rocky5~rocky1+rocky2+rocky3+rocky4, 
         data = trainingdata,linout=1,size = j,maxit = 10000),
    validationdata))^2)
}}
colMeans(modelperformance.nnt)[4] #0.9083277
which.min(colMeans(modelperformance.nnt)) #4

#----------------------------------        nnt(x^2, size=j)      -----------------------------------------------

modelperformance.nnt2 <- matrix(NA,nFold,50)
for (fold in 1:nFold) {
  for (j in 1:50){
    trainingdata <- subset(completeDB,valnum!=fold)
    validationdata <- subset(completeDB,valnum==fold)
    modelperformance.nnt2[fold,j] <- mean((validationdata$rocky5-predict(
      nnet(rocky5~rocky1^2+rocky2^2+rocky3^2+rocky4^2, 
           data = trainingdata,linout=1,size = j,maxit = 10000),
      validationdata))^2)
  }}
colMeans(modelperformance.nnt2)[3] #0.9108843
which.min(colMeans(modelperformance.nnt2)) #3

#----------------------------------        nnt(log(x), size=j)      -----------------------------------------------

modelperformance.nnt3 <- matrix(NA,nFold,50)
for (fold in 1:nFold) {
  for (j in 1:50){
    trainingdata <- subset(completeDB,valnum!=fold)
    validationdata <- subset(completeDB,valnum==fold)
    modelperformance.nnt3[fold,j] <- mean((validationdata$rocky5-exp(predict(
      nnet(log(rocky5)~log(rocky1)+log(rocky2)+log(rocky3)+log(rocky4), 
           data = trainingdata,linout=1,size = j,maxit = 10000),
      validationdata)))^2)
  }}
colMeans(modelperformance.nnt3)[6] #0.9406349
which.min(colMeans(modelperformance.nnt3)) #6

#----------------------------------        nnt(interactions, size=j)      -----------------------------------------------

modelperformance.nnt4 <- matrix(NA,nFold,50)
for (fold in 1:nFold) {
  for (j in 1:50){
    trainingdata <- subset(completeDB,valnum!=fold)
    validationdata <- subset(completeDB,valnum==fold)
    modelperformance.nnt4[fold,j] <- mean((validationdata$rocky5-exp(predict(
      nnet(rocky5~rocky1*rocky2+rocky1*rocky3+rocky1*rocky4+rocky1*rocky4+
        rocky2*rocky3+rocky2*rocky4+rocky3*rocky4+rocky1*rocky2*rocky3+rocky1*rocky2*rocky4+
        rocky2*rocky3*rocky4+rocky1*rocky2*rocky3*rocky4, 
           data = trainingdata,linout=1,size = j,maxit = 10000),
      validationdata)))^2)
  }}
colMeans(modelperformance.nnt4)[3] #1193.029
which.min(colMeans(modelperformance.nnt4)) #3

# apply best NNT model to entire dataset
bestNNT <- nnet(rocky5~rocky1+rocky2+rocky3+rocky4, 
                data = completeDB,linout=1,size = 4,maxit = 10000) # apply the best model to complete database

############################### Using test data set to predict Rocky5 ###############################################
testdb <- read.csv("/Users/jiadawei/Desktop/Market Analytics/Homework 2 - Test Set.csv") # load test data set
library(caret)
LinearRocky5 <- predict(bestlinear,testdb[,1:4])
testdb$LinearRocky5 <- LinearRocky5$`1413`
testdb$MARSRocky5 <- predict(bestMARS,testdb[,1:4])
testdb$nntRocky5 <- predict(bestNNT,testdb[,1:4])
testdb$kknnRocky5 <- kknn(formula = allModelsList[[31740]],train = completeDB, test = testdb, k = 10, distance = 1)$fitted.value

write.csv(testdb,file ="/Users/jiadawei/Desktop/Market Analytics/bestPredictions5.csv" )
save(completeDB,db,db.base, file="Team5Dataset.Rdata")
