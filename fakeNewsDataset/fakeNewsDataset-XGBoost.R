# https://gist.github.com/dkincaid/87f0fbeb912cf23816c340b4fbe30baa
# https://community.rstudio.com/t/document-term-matrix-in-xgboost-classifier/10043/12

library(xgboost)
library(caret)
library(tm)
library(text2vec)
library(pdp)
library(Matrix)




# =============================================================================== =
#  Loading the dataset ----
# =============================================================================== =
train_dataset <- read.csv("train.csv")
# train_dataset <- read.csv("/home/adelo/gofaaas/r_prog/4-Fake_news_dataset_using_RTextTools_and_XGBoost/train.csv")
# train_dataset <- read.csv("/home/adelo/1-system/1-disco_local/1-mis_archivos/1-pe/1-ciencia/1-computacion/2-data_analysis-machine_learning/gofaaaz-machine_learning/r_prog/4-Fake_news_dataset_using_RTextTools_and_XGBoost/train.csv",  nrows = 100)




# =============================================================================== =
#  Splitting the data into Train and Test data ----
# =============================================================================== =
dataset <- train_dataset

set.seed(16102016)                             # To fix the sample 

# Randomly Taking the 70% of the rows (70% records will be used for training  sampling without replacement. The remaining 30% will be used for testing)
samp_id = sample(1:nrow(dataset),              # do ?sample to examine the sample() func
                 round(nrow(dataset)*.70),
                 replace = F)
train = dataset[samp_id,]                      # 70% of training data set, examine struc of samp_id obj
test  = dataset[-samp_id,]                     # remaining 30% of training data set
dim(train) ; dim(test)


# Join the data sets
# Esto se hace porque cuando se ejecuta create_container, se especifica que parte de la data es «train» y que parte es «test»
data = rbind(train,test)




# =============================================================================== =
#  Text processing ----
# =============================================================================== =
data$text = tolower(data$text)      # Convert to lower case
text = data$text                    # Taking just the text column
text = removePunctuation(text)      # Remove punctuation marks
text = removeNumbers(text)          # Remove numbers
text = stripWhitespace(text)        # Remove blank space

text_train = text[1:nrow(train)]
text_test  = text[(nrow(train)+1):nrow(data)]


# Coded labels
labels_train = train$label
labels_test  = test$label




# =============================================================================== =
#  Tokenize the text and create a vocabulary of tokens including document counts ----
# =============================================================================== =

# ---------------------------------------------------------------- -
#   * Vocabulary of tokens for the train data ----
# ---------------------------------------------------------------- -
vocab_train <- create_vocabulary(itoken(text_train,
                                        preprocessor = tolower,
                                        tokenizer    = word_tokenizer))


# ---------------------------------------------------------------- -
#   * Vocabulary of tokens for the test data ----
# ---------------------------------------------------------------- -
vocab_test <- create_vocabulary(itoken(text_test,
                                       preprocessor = tolower,
                                       tokenizer    = word_tokenizer))




# =============================================================================== =
#  Build a document-term matrix using the tokenized review text. This returns a dgCMatrix object ----
# =============================================================================== =


# ---------------------------------------------------------------- -
#   * Document-term matrix for the train data ----
# ---------------------------------------------------------------- -
dtm_train <- create_dtm(itoken(text_train,
                               preprocessor = tolower,
                               tokenizer = word_tokenizer),
                        vocab_vectorizer(vocab_train))


# ---------------------------------------------------------------- -
#   * Document-term matrix for the test data ----
# ---------------------------------------------------------------- -
dtm_test <- create_dtm(itoken(text_test,
                              preprocessor = tolower,
                              tokenizer = word_tokenizer),
                       vocab_vectorizer(vocab_train))




# =============================================================================== =
#  Turn the DTM into an XGB matrix using the labels that are to be learned ----
# =============================================================================== =
train_matrix <- xgb.DMatrix(dtm_train, label = labels_train)




# =============================================================================== =
#  Xgboost model building ----
# =============================================================================== =
xgb_params = list(
  objective = "binary:logistic",
  eta = 0.01,
  max.depth = 5,
  eval_metric = "auc")


date()
xgb_fit <- xgboost(data = train_matrix, params = xgb_params, nrounds = 10000, print_every_n = 500)
date()




# =============================================================================== =
#  Confusion matrix ----
# =============================================================================== =
# This tutorial is very important to understant this part:
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html

# ---------------------------------------------------------------- -
#   * Confusion matrix using the train data ----
# ---------------------------------------------------------------- -
# Create our prediction probabilities
pred <- predict(xgb_fit, dtm_train)

# Set our cutoff 
pred.resp <- ifelse(pred >= 0.5, 1, 0)

# Create the confusion matrix
# https://rpubs.com/mharris/multiclass_xgboost
confusionMatrix(factor(pred.resp), factor(labels_train), positive="1")


# ---------------------------------------------------------------- -
#   * Confusion matrix using the test data ----
# ---------------------------------------------------------------- -
# Create our prediction probabilities
pred <- predict(xgb_fit, dtm_test)

# Set our cutoff 
pred.resp <- ifelse(pred >= 0.5, 1, 0)

# Create the confusion matrix
# https://rpubs.com/mharris/multiclass_xgboost
confusionMatrix(factor(pred.resp), factor(labels_test), positive="1")




# =============================================================================== =
#  Check the feature importance ----
# =============================================================================== =
importance_vars_train <- xgb.importance(model=xgb_fit, feature_names = colnames(train_matrix))
head(importance_vars_train, 20)




# =============================================================================== =
#  Try to plot a partial dependency plot of one of the features ----
# =============================================================================== =
# partial(xgb_fit, train = text, pred.var = "bad")
xgb.plot.importance(importance_vars_train, top_n = 20)
xgb.plot.importance(importance_vars_train, top_n = sum(importance_vars_train$Gain >= 0.05))
xgb.plot.importance(importance_vars_train, top_n = sum(cumsum(importance_vars_train$Gain) <= 0.85))


xgb.plot.importance(importance_vars_test, top_n = 20)
xgb.plot.importance(importance_vars_test, top_n = sum(importance_vars_test$Gain >= 0.05))
xgb.plot.importance(importance_vars_test, top_n = sum(cumsum(importance_vars_test$Gain) <= 0.85))


