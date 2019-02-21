# Set a seed for reproducibility
set.seed(0)

# Load the libs required for the analysis
library(class)

# Load the training and test datasets
train <- read.csv("C:/Users/user/Downloads/train.csv")
test <- read.csv("C:/Users/user/Downloads/test.csv")



# Extract the training set labels
trainlabel<-train[,1]
trainlabel<-data.frame(trainlabel)
trainlabel<-as.vector(trainlabel)
trainlabel<-as.numeric(t(trainlabel))

trainlebel<-as.vector(trainlabel)
head(trainlebel)

# Remove the label column, scale by the max value and then center the scaled data.
train.x<-train[,-1]/255
train.c<-scale(train.x,center=TRUE,scale=FALSE)

# Identify the feature means for the training data
# This can then be used to center the validation data
trainMeans<-colMeans(train.x)
trainMeansMatrix<-do.call("rbind",replicate(nrow(test),trainMeans,simplif=FALSE))

# Generate a covariance matrix for the centered training data
train.cov<-cov(train.x)

# Run a principal component analysis using the training correlation matrix
pca.train<-prcomp(train.cov)

# Identify the amount of variance explained by the PCs
varEx<-as.data.frame(pca.train$sdev^2/sum(pca.train$sdev^2))
varEx<-cbind(c(1:784),cumsum(varEx[,1]))
colnames(varEx)<-c("Nmbr PCs","Cum Var")
VarianceExplanation<-varEx[seq(0,200,10),]

# Because we can capture 99+% of the variation in the training data
# using only the first 50 PCs, we extract these for use in the KNN classifier
rotate<-pca.train$rotation[,1:50]
rotate[1,]

# Create the loading matrix based on the original training data
# This is the dimension reduction phase where we take the data
# matrix with 784 cols and convert it to a matrix with only 50 cols
dim(train.c)
dim(rotate)
trainFinal<-as.matrix(train.c)%*%(rotate)
length(trainFinal)
trainFinal<-data.frame(trainFinal)
summary(trainlabel)

# Spilliting data for prediction
trainlabel2<-trainlabel[4201:42000]
trainlabel1<-trainlabel[1:4200]
length(trainlabel1)
length(trainlabel2)

trainFinal2<-trainFinal[4201:42000,]
trainFinal1<-trainFinal[1:4200,]

# We then create a loading matrix for the testing data after applying
# the same centering and scaling convention as we did for training set
test.x<-test/255
testFinal<-as.matrix(test.x-trainMeansMatrix)%*%(rotate)
dim(testFinal)
testFinal<-data.frame(testFinal)

#use cv to decide best k for knn, need about an hour to run, don't perform without consideration
#create folds
folds<-createFolds(1:42000,k=10)
length(folds)
folds[[1]]
a<-trainFinal[ -folds[[1]],]
dim(a)

ks <- c(1,3,5,7,9,11,13,15)
res <- sapply(ks, function(k) {
  ##try out each version of k
  res.k <- sapply(seq_along(folds), function(i) {
    ##loop over each of the 10 cross-validation folds
    ##predict the held-out samples using k nearest neighbors
    pred <- knn(train=trainFinal[ -folds[[i]],],
                test=trainFinal[ folds[[i]], ],
                cl=t(trainlabel[ -folds[[i]] ]), k = k)
    ##the ratio of misclassified samples
    mean(trainlabel[ folds[[i]]] == pred)
  })
  ##average over the 10 folds
  mean(res.k)
})

plot(ks, res, type="o",ylab="accuracy")

# Run the KNN predictor on the dim reduced datasets
predict<-knn(train=trainFinal2,test=trainFinal1,cl=t(trainlabel2),k=3)
mean(predict==trainlabel1)











##optimal result:k=5,pc=50,97.57%

##########
#RUN A RANDOM FOREST BENCHMARK FOR COMPARISON
library(randomForest)
set.seed(0)
 
numTrain <- 10000
numTrees <- 25
rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]

rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
RF_predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
##########

# Output the results
cat("The PCA+KNN results are: ",sum(results)/length(results)," of the random forest results.","\n")
