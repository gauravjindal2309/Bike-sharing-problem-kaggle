# Bike-sharing-problem-kaggle
#dependent variables have natural outliers so we will predict log of dependent variables.
train <- read.csv('train.csv')
test <- read.csv('test.csv')

datetime = test$datetime
train_x = train
test_x = test
train_x$datetime = NULL
test_x$datetime = NULL
train_x$casual = NULL
train_x$registered = NULL

#linear
linear = lm(count~.,data =train_x)
summary(linear)
pred_linear = predict(linear, newdata =test_x)
Mysubmission = data.frame(datetime = datetime, count = pred_linear)
write.csv(Mysubmission,"Submission_linear.csv", row.names = FALSE)

RSME = sqrt(sum((sol$count-pred_linear)^2)/nrow(test_x))

# tree
# score = 1.38
require(rpart)
require(rpart.plot)
tree = rpart(count~.,data =train_x,minbucket=5,cp=.01)
prp(tree)
pred_tree=predict(tree,newdata = test_x)
Mysubmission = data.frame(datetime = datetime, count = pred_tree)
write.csv(Mysubmission,"Submission_tree.csv", row.names = FALSE)

#randomforest 
# score = 1.32814
require(randomForest)
forest=randomForest(count~.,data =train_x,ntree=100,nodesize=21,cp=.39)
pred_rf=predict(forest,newdata = test_x)
Mysubmission = data.frame(datetime = datetime, count = pred_rf)
write.csv(Mysubmission,"Submission_rf.csv", row.names = FALSE)

#SVM 
# score = 1167.79197
require(e1071)
svm=svm(count~.,data =train_x,kernel="radial")
pred_svm=predict(svm,newdata=test_x)
Mysubmission = data.frame(datetime = datetime, count = pred_svm)
write.csv(Mysubmission,"Submission_svm.csv", row.names = FALSE)

#XGB 
# score = 1427.58821
train_y =train_x
train_y$item_Outlet = NULL
label=train_x$count
label = as.numeric(label)
bst=xgboost(data = as.matrix(train_y),label = label,max.depth = 20, eta = 1, nthread = 4, nround = 50, objective = "reg:linear")
pred_xgb=predict(bst,as.matrix(test_x))
Mysubmission = data.frame(datetime = datetime, count = pred_xgb)
write.csv(Mysubmission,"Submission_xgb.csv", row.names = FALSE)
