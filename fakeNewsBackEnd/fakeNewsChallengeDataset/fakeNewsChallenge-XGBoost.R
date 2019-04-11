# https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
# https://stackoverflow.com/questions/1299871/how-to-join-merge-data-frames-inner-outer-left-right

library(xgboost)
library(caret)
library(tm)
library(text2vec)
library(pdp)
library(Matrix)
library(dplyr)




# =============================================================================== =
#  Loading the dataset ----
# =============================================================================== =
train_stances <- read.csv("/home/adelo/gofaaas/r_prog/FakeNewsChallenge/train_stances.csv")
train_bodies  <- read.csv("/home/adelo/gofaaas/r_prog/FakeNewsChallenge/train_bodies.csv")
# 
# train_stances <- read.csv("/home/adelo/1-system/desktop/it_cct/5-Applied_Technology_Group_Project/gofaaaz-machine_learning/r_prog/6-FakeNewsChallenge/train_stances.csv",  nrows = 1000)
# train_bodies  <- read.csv("/home/adelo/1-system/desktop/it_cct/5-Applied_Technology_Group_Project/gofaaaz-machine_learning/r_prog/6-FakeNewsChallenge/train_bodies.csv",  nrows = 1000)




# =============================================================================== =
#  Splitting the data into Train and Test data ----
# =============================================================================== =
dataset <- merge(train_stances, train_bodies)
dataset <- dataset[c("Body.ID", "articleBody", "Headline", "Stance")]

dataset = dataset[1:1000,]

set.seed(16102016)                             # To fix the sample 

# Randomly Taking the 70% of the rows (70% records will be used for training  sampling without replacement. The remaining 30% will be used for testing)
samp_id = sample(1:nrow(dataset),              # do ?sample to examine the sample() func
                 round(nrow(dataset)*.70),
                 replace = F)
train = dataset[samp_id,]                      # 70% of training data set, examine struc of samp_id obj
test  = dataset[-samp_id,]                     # remaining 30% of training data set
dim(train) ; dim(test)


# Join the data sets
data = rbind(train,test)




# =============================================================================== =
#  Text processing ----
# =============================================================================== =

# Text
text <- paste(data$articleBody, data$Headline)

text = tolower(text)            # Convert to lower case
text = removePunctuation(text)  # Remove punctuation marks
text = removeNumbers(text)      # Remove numbers
text = stripWhitespace(text)    # Remove blank space

text_train = text[1:nrow(train)]
text_test  = text[(nrow(train)+1):nrow(data)]


# Labels
lb <- as.numeric(data$Stance) - 1

num_class <- 4

labels_train = lb[1:nrow(train)]
labels_test  = lb[(nrow(train)+1):nrow(data)]




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
date()
xgb_fit <- xgboost(data = train_matrix,
                   max_depth = 4, eta = 0.5, nthread = 2, nrounds = 2000, print_every_n = 200, subsample = 0.5,
                   objective = "multi:softprob", num_class = num_class)
date()
date()




# =============================================================================== =
#  Making predictions from the model created and displaying a Confusion matrix ----
# =============================================================================== =

# ---------------------------------------------------------------- -
#   * Using the train data ----
# ---------------------------------------------------------------- -

# Predict for softmax returns num_class probability numbers per case:
pred <- predict(xgb_fit, dtm_train)


# Reshape it to a num_class-columns matrix
pred <- matrix(pred, ncol=num_class, byrow=TRUE)


# Convert the probabilities to softmax labels
pred_labels <- max.col(pred) - 1


# Error: It should result in the same error as seen in the last iteration
sum(pred_labels != labels_train)/length(labels_train)


# Confusion matrix
confusionMatrix(factor(pred_labels), factor(labels_train), positive="1")


# ---------------------------------------------------------------- -
#   * Using the test data ----
# ---------------------------------------------------------------- -

# Predict for softmax returns num_class probability numbers per case:
pred <- predict(xgb_fit, dtm_test)


# Reshape it to a num_class-columns matrix
pred <- matrix(pred, ncol=num_class, byrow=TRUE)


# Convert the probabilities to softmax labels
pred_labels <- max.col(pred) - 1


# Error: It should result in the same error as seen in the last iteration
sum(pred_labels != labels_test)/length(labels_test)


# Confusion matrix
confusionMatrix(factor(pred_labels), factor(labels_test), positive="1")



