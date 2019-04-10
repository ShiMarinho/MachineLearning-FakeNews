library(RTextTools)
library(tm)
library(base64enc)
library(qdapRegex)
library(dplyr)




# =============================================================================== =
#  Loading the dataset ----
# =============================================================================== =
train_dataset <- read.csv("train.csv")
train_dataset <- read.csv("/home/adelo/gofaaas/r_prog/4-Fake_news_dataset_using_RTextTools_and_XGBoost/train.csv")
# train_dataset <- read.csv("/home/adelo/1-system/1-disco_local/1-mis_archivos/1-pe/1-ciencia/1-computacion/2-data_analysis-machine_learning/gofaaaz-machine_learning/r_prog/4-Fake_news_dataset_using_RTextTools_and_XGBoost/train.csv",  nrows = 1000)




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
data$text = tolower(data$text)        # Convert to lower case
text = data$text                      # Taking just the text column
text = removePunctuation(text)        # Remove punctuation marks
text = removeNumbers(text)            # Remove numbers
text = stripWhitespace(text)          # Remove blank space


# Coded labels
labels = data$label




# =============================================================================== =
#  Creating the document term matrix ----
# =============================================================================== =
# https://en.wikipedia.org/wiki/Document-term_matrix
cor  = Corpus(VectorSource(text))           # Create text corpus
dtm  = DocumentTermMatrix(cor,              # Craete DTM
                          control = list(weighting =
                                          function(x)
                                            weightTfIdf(x, normalize = F)))  # IDF weighing
dim(dtm)


# Sorts a sparse matrix in triplet format (i,j,v) first by i, then by j  (https://groups.google.com/forum/#!topic/rtexttools-help/VILrGoRpRrU)
# Args:
#   working.dtm: a sparse matrix in i,j,v format using $i $j and $v respectively. Any other variables that may exist in the sparse matrix are not operated on, and will be returned as-is.
# Returns:
#   A sparse matrix sorted by i, then by j.
ResortDtm <- function(working.dtm) {
  working.df <- data.frame(i = working.dtm$i, j = working.dtm$j, v = working.dtm$v)  # create a data frame comprised of i,j,v values from the sparse matrix passed in.
  working.df <- working.df[order(working.df$i, working.df$j), ]                      # sort the data frame first by i, then by j.
  working.dtm$i <- working.df$i                                                      # reassign the sparse matrix' i values with the i values from the sorted data frame.
  working.dtm$j <- working.df$j                                                      # ditto for j values.
  working.dtm$v <- working.df$v                                                      # ditto for v values.
  return(working.dtm)                                                                # pass back the (now sorted) data frame.
}

dtm <- ResortDtm(dtm)




# =============================================================================== =
#  Creating the model ----
# =============================================================================== =

# ---------------------------------------------------------------- -
#   * Creating the 'container' obj ----
# ---------------------------------------------------------------- -
# Creates a 'container' obj for training, classifying, and analyzing docs:
container <- create_container(dtm,
                              t(labels),                                     # Labels or the Y variable / outcome we want to train on
                              trainSize = 1:nrow(train),
                              testSize  = (nrow(train)+1):nrow(data),
                              virgin    = FALSE)                             # Whether to treat the classification data as 'virgin' data or not
                                                                             # If virgin = TRUE, then machine won;t borrow from prior datasets
# View struc of the container obj; is a list of training n test data
str(container)


# ---------------------------------------------------------------- -
#   * Creating the 'model' obj ----
# ---------------------------------------------------------------- -
# https://journal.r-project.org/archive/2013/RJ-2013-001/RJ-2013-001.pdf
# "GLMNET", "SVM", "MAXENT", "RF", "TREE", "BOOSTING", "NNET", "BAGGING", "SLDA"
date()
models  <- train_models(container,
                        algorithms=c("SLDA"))
date()




# =============================================================================== =
#  Makes predictions from the model created ----
# =============================================================================== =
# Makes predictions from a train_models() object
results <- classify_models(container, models)
dim(results)
head(results)




# =============================================================================== =
#  Determining the accuracy of prediction results ----
# =============================================================================== =
# Building a confusion matrix
out = data.frame(model_prob   = results$TREE_PROB,
                 model_label  = results$TREE_LABEL,
                 actual_label = data$label[(nrow(train)+1):nrow(data)])        # Actual value of Y

dim(out);
head(out);


summary(out)
# --------------------------------------------------
(z = as.matrix(table(out[,2], out[,3])))           # Display the confusion matrix
# --------------------------------------------------
(pct = round(((z[1,1] + z[2,2])/sum(z))*100, 2))   # Prediction accuracy in % terms






