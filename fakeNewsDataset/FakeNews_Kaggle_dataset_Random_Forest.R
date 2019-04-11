#src: https://journal.r-project.org/archive/2013/RJ-2013-001/RJ-2013-001.pdf
#article: RTextTools: A Supervised Learning Package for Text Classification
#libraries

library(RTextTools)
library(tidytext)
library(stringr)
library(tm)

# LOAD train and test fake news dataset 
train_dataset <- read.csv("/Users/ms/Downloads/fake-news/train.csv")
test_dataset  <- read.csv("~/Downloads/fake-news/test.csv")

#------------------------------Step 1---------------------------------------------
# 1.a PREPROCESS - data preparation, importing, cleaning and general preprocessing
# removing: numbers, punctuation, sparseTerms(basically 0s), stop words(the, and, it)
# mutating all text to lower case
train_dataset$text = tolower(train_dataset$text)
# 1.b CREATING THE DOCUMENT-TERM MATRIX

matrix <- create_matrix(train_dataset$text ,language="english", 
                        removeNumbers=TRUE, stemWords=FALSE, removePunctuation = TRUE, removeSparseTerms = .998, 
                        removeStopwords = TRUE, stripWhitespace = TRUE)


#------------------------------Step 2---------------------------------------------
# 2. CREATING A CONTAINER to output a matrix_container within train and test sparse matrices, corresponding
# vectors of train and test codes, and character vector of trm label names.
dim(train_dataset)
#output 
#[1] 20800     5
dim(test_dataset)
#output 
#[1] 5200    4
container <- create_container(matrix, train_dataset$label, trainSize = 1:16640, testSize = 16641:20800, virgin=FALSE)
str(container)
#output
#Formal class 'matrix_container' [package "RTextTools"] with 6 slots
#..@ training_matrix      :Formal class 'matrix.csr' [package "SparseM"] with 4 slots
#.. .. ..@ ra       : num [1:4048493] 1 3 1 1 1 1 5 3 2 1 ...
#.. .. ..@ ja       : int [1:4048493] 385 421 474 493 560 639 669 781 785 829 ...
#.. .. ..@ ia       : int [1:16641] 1 259 516 948 1111 1173 1264 1435 1729 2132 ...
#.. .. ..@ dimension: int [1:2] 16640 14938
#..@ classification_matrix:Formal class 'matrix.csr' [package "SparseM"] with 4 slots
#.. .. ..@ ra       : num [1:987274] 1 1 1 2 1 1 5 1 1 1 ...
#.. .. ..@ ja       : int [1:987274] 405 722 785 992 1095 1121 1523 1541 1617 1646 ...
#.. .. ..@ ia       : int [1:4161] 1 170 547 553 1005 1410 1429 1509 1530 1872 ...
#.. .. ..@ dimension: int [1:2] 4160 14938
#..@ training_codes       : Factor w/ 2 levels "0","1": 2 1 2 2 2 1 2 1 1 1 ...
#..@ testing_codes        : Factor w/ 2 levels "0","1": 2 1 2 1 1 2 1 2 1 2 ...
#..@ column_names         : chr [1:14938] "–" "—" "…" "…”" ...
#..@ virgin               : logi FALSE

summary(container)
#output
#  Length            Class             Mode 
#1 matrix_container               S4 


#------------------------------Step 3---------------------------------------------
#3. Training models
RF <- train_model(container,"RF")



#------------------------------Step 4---------------------------------------------
#4. Classifying data using trained models
RF_CLASSIFY <- classify_model(container, RF)


#------------------------------Step 5---------------------------------------------
#5. Analytics
analytics_rf <- create_analytics(container, RF_CLASSIFY)



#------------------------------Step 6---------------------------------------------
#6. Testing algorithm accurary
summary(analytics_rf)
#output
#ENSEMBLE SUMMARY

#n-ENSEMBLE COVERAGE n-ENSEMBLE RECALL
#n >= 1                   1              0.96


#ALGORITHM PERFORMANCE

#FORESTS_PRECISION    FORESTS_RECALL    FORESTS_FSCORE 
#           0.960             0.965             0.960 



# CREATE THE data.frame SUMMARIES
topic_summary <- analytics_rf@label_summary
#output
#NUM_MANUALLY_CODED NUM_CONSENSUS_CODED NUM_PROBABILITY_CODED PCT_CONSENSUS_CODED
#0               2047                2076                  2076           101.41671
#1               2113                2084                  2084            98.62754
#PCT_PROBABILITY_CODED PCT_CORRECTLY_CODED_CONSENSUS PCT_CORRECTLY_CODED_PROBABILITY
#0             101.41671                      96.77577                        96.77577
#1              98.62754                      95.50402                        95.50402

alg_summary <- analytics_rf@algorithm_summary
#output
#FORESTS_PRECISION FORESTS_RECALL FORESTS_FSCORE
#0              0.95           0.97           0.96
#1              0.97           0.96           0.96

ens_summary <-analytics_rf@ensemble_summary
#output
# n-ENSEMBLE COVERAGE n-ENSEMBLE RECALL
#n >= 1                   1              0.96

doc_summary <- analytics_rf@document_summary
#output
#FORESTS_LABEL FORESTS_PROB MANUAL_CODE CONSENSUS_CODE CONSENSUS_AGREE CONSENSUS_INCORRECT
#1               1        0.740           1              1               1                   0
#2               0        0.660           0              0               1                   0
#3               1        1.000           1              1               1                   0
#4               0        0.880           0              0               1                   0
#5               0        0.865           0              0               1                   0
#6               1        0.975           1              1               1                   0
#7               0        0.540           0              0               1                   0
#9               0        0.865           0              0               1                   0
#10              1        0.795           1              1               1                   0
# #11              0        0.840           0              0               1                   0
# #12              0        0.615           0              0               1                   0
# #13              0        0.820           0              0               1                   0
# #14              1        0.960           1              1               1                   0
# #15              1        0.975           1              1               1                   0
# #16              1        0.815           1              1               1                   0
# #17              0        0.795           0              0               1                   0
# #18              1        0.915           1              1               1                   0
# #19              0        0.845           0              0               1                   0
# #20              1        0.830           1              1               1                   0
# 21              1        1.000           1              1               1                   0
# 22              0        0.750           0              0               1                   0
# 23              1        0.705           1              1               1                   0
# 24              0        0.855           0              0               1                   0
# 25              0        0.780           0              0               1                   0
# 26              0        0.830           0              0               1                   0
# 27              1        0.670           1              1               1                   0
# 28              1        0.600           1              1               1                   0
# 29              1        0.550           1              1               1                   0
# 30              0        0.930           0              0               1                   0
# 31              0        0.625           0              0               1                   0
# 32              1        0.675           1              1               1                   0
# 33              0        0.770           0              0               1                   0
# 34              1        0.950           1              1               1                   0
# 35              0        0.865           0              0               1                   0
# 36              1        0.755           1              1               1                   0
# 37              0        0.860           0              0               1                   0
# 38              0        0.685           0              0               1                   0
# 39              0        0.820           0              0               1                   0
# 40              1        0.795           1              1               1                   0
# 41              0        0.780           0              0               1                   0
# 42              0        0.775           0              0               1                   0
# 43              1        0.640           1              1               1                   0
# 44              1        0.885           1              1               1                   0
# 45              0        0.530           1              0               1                   1
# 46              1        0.720           1              1               1                   0
# 47              0        0.880           0              0               1                   0
# 48              1        0.825           1              1               1                   0
# 49              1        0.990           1              1               1                   0
# 50              0        0.790           0              0               1                   0
# 51              0        0.860           0              0               1                   0
# 52              0        0.780           0              0               1                   0
# 53              0        0.860           0              0               1                   0
# 54              0        0.890           0              0               1                   0
# 55              0        0.830           0              0               1                   0
# 56              0        0.920           0              0               1                   0
# 57              0        0.895           0              0               1                   0
# 58              1        0.515           0              1               1                   1
# 59              1        0.685           1              1               1                   0
# 60              0        0.850           0              0               1                   0
# 61              1        0.960           1              1               1                   0
# 62              1        0.980           1              1               1                   0
# 63              1        0.995           1              1               1                   0
# 64              1        0.800           1              1               1                   0
# 65              0        0.900           0              0               1                   0
# 66              0        0.775           0              0               1                   0
# 67              0        0.875           0              0               1                   0
# 68              1        0.800           1              1               1                   0
# 69              0        0.505           1              0               1                   1
# 70              0        0.835           0              0               1                   0
# 71              0        0.615           0              0               1                   0
# 72              1        0.705           1              1               1                   0
# 73              1        0.635           1              1               1                   0
# 74              1        0.735           1              1               1                   0
# 75              0        0.710           0              0               1                   0
# 76              1        1.000           1              1               1                   0
# 77              1        0.615           1              1               1                   0
# 78              1        0.835           1              1               1                   0
# 79              0        0.780           0              0               1                   0
# 80              0        0.630           0              0               1                   0
# 81              0        0.920           0              0               1                   0
# 82              1        0.880           1              1               1                   0
# 83              1        0.805           1              1               1                   0
# 84              0        0.600           0              0               1                   0
# 85              0        0.825           0              0               1                   0
# 86              0        0.860           0              0               1                   0
# 87              0        0.810           0              0               1                   0
# 88              1        0.685           1              1               1                   0
# 89              1        1.000           1              1               1                   0
# 90              1        0.780           1              1               1                   0
# 91              1        0.995           1              1               1                   0
# 92              1        0.940           1              1               1                   0
# 93              1        0.635           1              1               1                   0
# 94              1        0.870           1              1               1                   0
# 95              1        0.760           0              1               1                   1
# 96              0        0.795           0              0               1                   0
# 97              1        0.720           1              1               1                   0
# 98              1        0.540           1              1               1                   0
# 99              1        0.955           1              1               1                   0
# 100             0        0.855           0              0               1                   0
# 101             0        0.805           0              0               1                   0
# 102             1        0.620           1              1               1                   0
# 103             0        0.755           0              0               1                   0
# 104             1        0.885           1              1               1                   0
# 105             1        0.695           1              1               1                   0
# 106             1        0.675           1              1               1                   0
# 107             0        0.760           0              0               1                   0
# 108             1        0.660           1              1               1                   0
# 109             0        0.885           0              0               1                   0
# 110             0        0.830           0              0               1                   0
# 111             0        0.720           0              0               1                   0
# 112             0        0.850           0              0               1                   0
# 113             0        0.885           0              0               1                   0
# 114             0        0.865           0              0               1                   0
# 115             0        0.885           0              0               1                   0
# 116             1        0.965           1              1               1                   0
# 117             0        0.775           0              0               1                   0
# 118             0        0.825           0              0               1                   0
# 119             1        0.850           1              1               1                   0
# 120             1        0.825           1              1               1                   0
# 121             0        0.830           0              0               1                   0
# 122             0        0.905           0              0               1                   0
# 123             1        0.940           1              1               1                   0
# 124             1        0.625           1              1               1                   0
# 125             0        0.830           0              0               1                   0
# #PROBABILITY_CODE PROBABILITY_INCORRECT
# #1                  1                     0
# #2                  0                     0
# #3                  1                     0
# #4                  0                     0
# #5                  0                     0
# #6                  1                     0
# #7                  0                     0
# #8                  1                     0
# #9                  0                     0
# #10                 1                     0
# #11                 0                     0
# #12                 0                     0
# 13                 0                     0
# 14                 1                     0
# 15                 1                     0
# 16                 1                     0
# 17                 0                     0
# 18                 1                     0
# 19                 0                     0
# 20                 1                     0
# 21                 1                     0
# 22                 0                     0
# 23                 1                     0
# 24                 0                     0
# 25                 0                     0
# 26                 0                     0
# 27                 1                     0
# 28                 1                     0
# 29                 1                     0
# 30                 0                     0
# 31                 0                     0
# 32                 1                     0
# 33                 0                     0
# 34                 1                     0
# 35                 0                     0
# 36                 1                     0
# 37                 0                     0
# 38                 0                     0
# 39                 0                     0
# 40                 1                     0
# 41                 0                     0
# 42                 0                     0
# 43                 1                     0
# 44                 1                     0
# 45                 0                     1
# 46                 1                     0
# 47                 0                     0
# 48                 1                     0
# 49                 1                     0
# 50                 0                     0
# 51                 0                     0
# 52                 0                     0
# 53                 0                     0
# 54                 0                     0
# 55                 0                     0
# 56                 0                     0
# 57                 0                     0
# 58                 1                     1
# 59                 1                     0
# 60                 0                     0
# 61                 1                     0
# 62                 1                     0
# 63                 1                     0
# 64                 1                     0
# 65                 0                     0
# 66                 0                     0
# 67                 0                     0
# 68                 1                     0
# 69                 0                     1
# 70                 0                     0
# 71                 0                     0
# 72                 1                     0
# 73                 1                     0
# 74                 1                     0
# 75                 0                     0
#76                 1                     0
#77                 1                     0
#78                 1                     0
#79                 0                     0
#80                 0                     0
#81                 0                     0
#82                 1                     0
#83                 1                     0
#84                 0                     0
#85                 0                     0
#86                 0                     0
#87                 0                     0
#88                 1                     0
#89                 1                     0
#90                 1                     0
#91                 1                     0
#92                 1                     0
#93                 1                     0
#94                 1                     0
#95                 1                     1
#96                 0                     0
#97                 1                     0
#98                 1                     0
#99                 1                     0
#100                0                     0
#101                0                     0
#102                1                     0
#103                0                     0
#104                1                     0
#105                1                     0
#106                1                     0
#107                0                     0
#108                1                     0
#109                0                     0
#110                0                     0
#111                0                     0
#112                0                     0
#113                0                     0
#114                0                     0
#115                0                     0
#116                1                     0
#117                0                     0
#118                0                     0
#119                1                     0
#120                1                     0
#121                0                     0
#122                0                     0
#123                1                     0
#124                1                     0
#125                0                     0
#[ reached 'max' / getOption("max.print") -- omitted 4035 rows ]



#------------------------------Step 7---------------------------------------------
#7 Ensemble agreement
create_ensembleSummary(analytics_rf@document_summary)
#output
# n-ENSEMBLE COVERAGE n-ENSEMBLE RECALL
#n >= 1                   1              0.96
