## Part a) Data Loading of Bank data
df1<-read.csv("F:\\Aegis\\Machine Learning\\Topic 8\\mushrooms.csv")
head(df1)
View(df1)
summary(df1)
str(df1)
colnames(df1)


## Part b) Appropriate Methods for Significant Variables
## removing columns having just one levels.
df1<-df1[!names(df1) %in% c("gill.attachment","veil.type","veil.color")]


## For significant variables:( all variables are categorical)
iv<-iv.mult(df1,"class",TRUE)
iv.plot.summary(iv)

## From the IV plot, least significant are cap.surface, ring.number, cap.shape, stalk.color.above.ring,
## stalk.root, population, gill.size, gill.spacing.

## Using Random Forest to verify and determine significant variables.
library(randomForest)
rf<-randomForest(class~.,data = df1)
rf$importance
colnames(df)

## Removing the columns:
df1<-df1[!names(df1) %in% c("cap.surface", "ring.number", "cap.shape", "stalk.color.above.ring","stalk.root", "population", "gill.size", "gill.spacing")]


## Part c) Dividing data into Development(train) and Validation(test) data.
library(caTools)
set.seed(1)
x1<-sample.split(df1$class,SplitRatio = 0.7)
train1<-subset(df1,x1==T)
test1<-subset(df1,x1==F)


## Part d) SVM model with linear kernel and Accuracy with Test data.
library(e1071)
model1=svm(class~.,data = train1,kernel="linear")
model1
summary(model1)
results1<-predict(model1,test1)
table(results1,test1$class)
mean(results1==test1$class) # accuracy

library(caret)
?precision()


## Part e) SVM with radial kernel, Tuning the model, and Accuracy with Test data.
## tuning the radial svm model.
model_tune1<-tune(svm,class~.,data = train1,kernel="radial",ranges = list(cost=c(0.001,0.01,0.1,1,5),gamma=c(0.001,0.01,0.1,1,5)))
summary(model_tune1)
bestmodel1= model_tune1$best.model
summary(bestmodel1)

## best svm model has cost 1 and gamma 0.1

## SVM with radial kernel
model_r1<-svm(class~.,data = train1,kernel="radial",cost=1,gamma=0.1)
model_r1
results_r1<-predict(model_r1,test1)
table(results_r1,test1$class)
mean(results_r1==test1$class)


## Part f) Naive Bayes Algorithm , and compare results with SVM.
### Naive-Bayes Model
model_nb1<-naiveBayes(class~.,data = train1)
model_nb1
results_nb1<-predict(model_nb1,test1)
table(results_nb1,test1$class)
mean(results_nb1==test1$class)

### For the mushrooms dataset, SVM with kernel goves 100% accuracy, whereas Naive-Bayes
## model gives 94.42% accuracy.
