#https://rpubs.com/Seun/455974

# =============================================================================== =
#  Setting up the enviroment ----
# =============================================================================== =
## Load required libraries
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer) 
library(e1071)         #For Naive Bayes
library(caret)         #For the Confusion Matrix
library(ggplot2)

# =============================================================================== =
#  Reading the data ----
# =============================================================================== =

data <- read.csv("fakeNewsBackEnd/manualNewsDataset/goofaasDataset.csv")
#View the first few lines of the dataset
head(data)
View(data)
dim(data)

#Find the proportions of unreliable from reliable news

train <- read.csv("fakeNewsBackEnd/kaggleFakeNewsDataset/train.csv")
#View the first few lines of the dataset
head(data)
View(data)
dim(data)
#Find the proportions of reliable vs unreliable news
table(data$label)
prop.table(table(data$label))


# =============================================================================== =
#  Data Processing ----
# =============================================================================== =

## Convert the label column from Character strings to factor. 
data$label <- factor(data$label)
prop.table(table(data$label))

## CLEANNING THE DATA ##
## The VectorSource() function will create one document for each sms text message. 
## The Vcorpus() function to create a volatile corpus from these individual text messages.

dataCorpus <- VCorpus(VectorSource(data$text))

data_dtm <- DocumentTermMatrix(dataCorpus, control = 
                                 list(tolower = TRUE, #Converts to lowecase
                                      removeNumbers = TRUE, #Removes numbers
                                      stopwords = TRUE, #Removes stop words
                                      removePunctuation = TRUE, #Removes punctuation
                                      stemming = TRUE)) #Applying stemming(involves trimming words suchs calling, called and calls to call)

dim(data_dtm)
data_dtm

## remove all terms in the corpus whose sparsity is greater than 90%.
data_dtm = removeSparseTerms(data_dtm, 0.90)
dim(data_dtm)

inspect(data_dtm[40:50, 10:15])

## Converting the word frequencies to Yes and No Labels ##
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

##  Apply the convert_count function to get final training and testing DTMs ##
dataNB <- apply(data_dtm, 2, convert_count)
dataset = as.data.frame(as.matrix(dataNB))

# =============================================================================== =
#  Descriptive and Exploratory Analysis of the data ----
# =============================================================================== =

## Building Word Frequency
freq<- sort(colSums(as.matrix(data_dtm)), decreasing=TRUE)
tail(freq, 10)

#identifying terms that appears frequently
findFreqTerms(data_dtm, lowfreq=60) 

## Plotting Word Frequency
wf<- data.frame(word=names(freq), freq=freq)
head(wf)

#  Building Word Cloud
set.seed(1234)
wordcloud(words = wf$word, freq = wf$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


# Adding the label variable to the Dataset
# The text data has been cleaned and now ready to be added to 
#the response variable “label” for the purpose of predictive analytics.
dataset$label = data$label


# =============================================================================== =
#  Data Analysis & Model Building ----
# =============================================================================== =

set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.80,0.20),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

# Check the proportion of the data split.
prop.table(table(train_set$label))
prop.table(table(test_set$label))

# =============================================================================== =
#  Model Fitting ----
# =============================================================================== =

## Naive Bayes Classifier ##

## predictions with naive bayes

control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time(classifier_nb <- naiveBayes(train_set, train_set$label, laplace = 1,
                                        trControl = control,tuneLength = 7) )


nb_pred = predict(classifier_nb, type = 'class', newdata = test_set)

confusionMatrix(nb_pred,test_set$label)

##using cross validation
control2 <- trainControl(method="cv", 10)

sms_model1 <- train(train_set, train_set$label, method="nb",
                    trControl=control2)
sms_model1

sms_model1_predict= predict(sms_model1, test_set)

confusionMatrix(sms_model1_predict, test_set$label)
