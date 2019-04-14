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

train_bodies <- read.csv("fakeNewsBackEnd/fakeNewsChallengeDataset/train_bodies.csv")
train_stances <- read.csv("fakeNewsBackEnd/fakeNewsChallengeDataset/train_stances.csv")

dataset <- merge(train_stances, train_bodies)
View(dataset)
dim(dataset)
#View the first few lines of the dataset
head(dataset)

#Find the proportions of agree, disgree, discuss, unrelated
table(dataset$Stance)
prop.table(table(dataset$Stance))

# =============================================================================== =
#  Data Processing ----
# =============================================================================== =

## Convert the label column from Character strings to factor. 
dataset$Stance <- factor(dataset$Stance)
prop.table(table(dataset$Stance))

## CLEANNING THE DATA ##
## The VectorSource() function will create one document for each sms text message. 
## The Vcorpus() function to create a volatile corpus from these individual text messages.
dataCorpus <- VCorpus(VectorSource(dataset$articleBody))


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

## Converting the word frequencies to Yes and No Labels ##
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

##  Apply the convert_count function to get final training and testing DTMs ##
dataNB <- apply(data_dtm, 2, convert_count)
data = as.data.frame(as.matrix(dataNB))

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
data$Stance = dataset$Stance


# =============================================================================== =
#  Data Analysis & Model Building ----
# =============================================================================== =

set.seed(222)
split = sample(2,nrow(data),prob = c(0.80,0.20),replace = TRUE)
train_set = data[split == 1,]
test_set = data[split == 2,] 

# Check the proportion of the data split.
prop.table(table(train_set$Stance))
prop.table(table(test_set$Stance))

# =============================================================================== =
#  Model Fitting ----
# =============================================================================== =

## Naive Bayes Classifier ##

## predictions with naive bayes

control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time( classifier_nb <- naiveBayes(train_set, train_set$Stance, laplace = 1,
                                         trControl = control,tuneLength = 7) )


nb_pred = predict(classifier_nb, type = 'class', newdata = test_set)

confusionMatrix(nb_pred,test_set$Stance)

