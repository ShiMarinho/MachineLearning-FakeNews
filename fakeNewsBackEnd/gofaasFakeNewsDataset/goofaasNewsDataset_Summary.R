# gofaas dataset description
# total fof 510 entries
# 304 - 0 (60%) - reliable 
# 206 - 1 (40%) - unreliable

#libraries
library(RTextTools)
library(tidytext)
library(stringr)
library(tm)

# LOAD train and test fake news dataset 
goofaasDataset <- read.csv("~/R_Files/fake-newsDataset/goofaasDataset.csv")


#------------------------------Step 1---------------------------------------------
# 1.a PREPROCESS - data preparation, importing, cleaning and general preprocessing
# removing: numbers, punctuation, sparseTerms(basically 0s), stop words(the, and, it)
# mutating all text to lower case
goofaasDataset$text = tolower(goofaasDataset$text)
# 1.b CREATING THE DOCUMENT-TERM MATRIX

matrix <- create_matrix(goofaasDataset$text ,language="english", 
                        removeNumbers=TRUE, stemWords=FALSE, removePunctuation = TRUE, removeSparseTerms = .998, 
                        removeStopwords = TRUE, stripWhitespace = TRUE)

#------------------------------Step 2---------------------------------------------
# 2. CREATING A CONTAINER to output a matrix_container within train and test sparse matrices, corresponding
# vectors of train and test codes, and character vector of trm label names.
dim(goofaasDataset)

container <- create_container(matrix, goofaasDataset$label, trainSize = 1:408, testSize = 409:511, virgin=FALSE)
str(container)


summary(container)

#------------------------------Step 3---------------------------------------------
#3. Training models
RF <- train_model(container,"RF")
TREE <- train_model(container,"TREE")
SVM <- train_model(container,"SVM")
MAXENT <- train_model(container,"MAXENT")
BOOSTING <- train_model(container,"BOOSTING")



#------------------------------Step 4---------------------------------------------
#4. Classifying data using trained models
RF_CLASSIFY <- classify_model(container, RF)
TREE_CLASSIFY <- classify_model(container, TREE)
SVM_CLASSIFY <- classify_model(container, SVM)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)


#------------------------------Step 5---------------------------------------------
#5. Analytics

analytics <- create_analytics(container,
                              cbind(RF_CLASSIFY,SVM_CLASSIFY,
                                    BOOSTING_CLASSIFY,
                                    RF_CLASSIFY, 
                                    TREE_CLASSIFY,
                                    MAXENT_CLASSIFY))



#------------------------------Step 6---------------------------------------------
#6. Testing algorithm accurary

summary(analytics)


# CREATE THE data.frame SUMMARIES

topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary


#------------------------------Step 7---------------------------------------------
#7 Ensemble agreement
create_ensembleSummary(analytics@document_summary)

RF <- cross_validate(container, 4, "RF")
TREE <- cross_validate(container, 4, "TREE")
SVM <- cross_validate(container, 4, "SVM")
MAXENT <- cross_validate(container, 4, "MAXENT")
BOOSTING <- cross_validate(container, 4, "BOOSTING")
NNET <- cross_validate(container, 4, "NNET")

#9 Exporting data


#---------other models --------------
write.csv(analytics@document_summary, "DocumentSummary.csv")


