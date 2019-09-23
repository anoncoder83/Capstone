# initial setup
library(dplyr)
library(tm)
library(lexicon)
library(stringr)
library(quanteda)
library(ggplot2)
library(R.utils)
library(dplyr)
library(tidyr)
library(caret)
#setwd("C:\\Data Science Projects\\JHU Course\\Submissions\\Course 10 Capstone Project")


## STEP 1 - Download data
#***********************************************************************************

#https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip

downloadURL <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"
dataFile <- "SwiftKeyData.zip"
if(!file.exists(dataFile))
  download.file(downloadURL, destfile=dataFile, method="curl")
unzip(dataFile)

## STEP 2 - Initial exploration
#***********************************************************************************

bloglines <- countLines("final\\en_US\\en_US.blogs.txt")
newslines <- countLines("final\\en_US\\en_US.news.txt")
twitterlines <- countLines("final\\en_US\\en_US.twitter.txt")
### Lines in each set - A line is counted as an element, and can contain many sentences.

print(paste("Lines in Blog data:",bloglines))
print(paste("Lines in news data:",newslines))
print(paste("Lines in twitter data:",twitterlines))


#As we see the corpora is quite big, for next steps, we will randomly subset the data and read only 15% of original data.

### Read Data Samples  

subsetWithBinom <- function(con, p, totalLines) {
  subset = c("");
  linecount <- 0
  
  while(linecount < totalLines){
    num<- 100;
    
    if(linecount+ 100 > totalLines){num <- totalLines - linecount;}
    
    dataline  <- readLines(con,n=num , encoding = "UTF-8") 
    if(rbinom(1,1, p)>0){
      subset <- c(subset , dataline);
    }
    linecount <- linecount + num;
  }
  return(subset)
}

set.seed(123)
con1 <- file("final\\en_US\\en_US.blogs.txt",open="r")
blogData <- subsetWithBinom(con1, p = 0.01, totalLines = bloglines)
close(con1)

set.seed(123)
con2 <- file("final\\en_US\\en_US.news.txt", open="r")
newsData <- subsetWithBinom(con2, p = 0.01, totalLines = newslines)
close(con2)

set.seed(123)
con3 <- file("final\\en_US\\en_US.twitter.txt", open="r")
twitterData <- subsetWithBinom(con3, p = 0.01, totalLines = twitterlines)
close(con3)

allData <- c(blogData,newsData,twitterData)
rm(blogData)
rm(newsData)
rm(twitterData)

inTrain <- rbinom(length(allData),1, p=0.75)
TrainData <- allData[which(inTrain == 1, arr.ind = T)]
TestData <- allData[which(inTrain == 0, arr.ind = T)]
#rm(allData)

## STEP 3 - Cleaning Data
#***********************************************************************************
#We can do further cleaning of text using lemmatization (Removing infliction of words). 

# Functions defined here. 
cleanData <- function(dat){
  dat <- tm::removeNumbers(dat)
  dat <- tm::removePunctuation(dat)
  dat <- tolower(dat)
  dat <- tm::stripWhitespace(dat)
  #todo : more cleaning
  pos <- lexicon::pos_preposition
  profanity <- as.data.frame(lexicon::profanity_alvarez)
  rem <- c(stopwords("en")," T ")
  dat <- tm::removeWords(dat,profanity$profanity)
  dat <- stringr::str_replace_all(dat,"^a-zA-Z\\s"," ")
  return(dat)
  
}

TrainData <- cleanData(TrainData)
TestData <- cleanData(TestData)

## STEP 4 - Explore the corpus
#***********************************************************************************

# I will use quanteda package here. I found it very convinient and straightforward once you understand the concepts of document-term matrix and tokens. Other packages to consider are Tokenizers and tm.
# generate tokens and frequencies of words
toks <- quanteda::tokens(TrainData, what = "word")
m1 <- dfm(toks) 
f1 <- textstat_frequency(m1)
toks_bigram <- tokens_ngrams(toks, n= 2)
m2 <- dfm(toks_bigram)
f2 <- textstat_frequency(m2)
toks_3gram <- tokens_ngrams(toks, n= 3)
m3 <- dfm(toks_3gram)
f3 <- textstat_frequency(m3)
f3 <- f3 %>% arrange(desc(frequency))
f3 <- f3[1:100000,]

rm(toks)
rm(toks_bigram)
rm(toks_3gram)
rm(m1)
rm(m2)
rm(m3)

# Let us look at top 5 rows of bigram and 3-gram in tabular format.
head(f1)
head(f2)
head(f3)

#*************** BUILDING MODELS *********************


#reference : http://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf
#reference : https://en.wikipedia.org/wiki/Katz%27s_back-off_model 

# 1. calculate 1-gram
#P(w_x) = count(wx)+1/N+V # where N is total token numbers , V = number of unique words
#(with smoothing by adding 1 to bothnumerator and denominator)

V = nrow(f1)
N <- sum(f1$frequency)

f1 <- f1 %>% mutate("prob" = (frequency +1)/(N+V))


# 2. and bigram frequencies use conditional probability 
# P(A|B) =  P(A,B) + 1/(P(B)+V)
#f2 <- f2 %>% mutate("prob" = (frequency + 1)/(f1$prob? + V))

bigrambyrow <- function(row)
{ 
    words <- unlist(strsplit(row[1],"_"))
    #print(words)
    freq<- as.numeric(row[2])
    #print(freq)
    found <- f1[f1$feature == words[2],]
    prewordfreq = 1 # defaulting to 1 in case its not in corpus # how can that happen
    if(nrow(found>0))
    {prewordfreq <- as.numeric(found$frequency)}
    
    #print(prewordfreq)
    p <- freq/( prewordfreq + V)
  return(p)
}

#sapply(f2,bigrambyrow)
bigramp <- apply(f2, 1, bigrambyrow)
bigramp <- as.numeric(bigramp)
f2 <- f2 %>% mutate("prob" = bigramp)

fun1 <- function(row){
  w <-unlist(strsplit(row[1],"_"))
  return(c(w[1]))
}

first <- apply(f2, 1, fun1)
second <- apply(f2, 1, 
                function(row){w <-unlist(strsplit(row[1],"_"));return(c(w[2]))})

f2$first <- first
f2$second <- second

# prediction
word <- "first"
preds <- f2 %>% filter(first == word) %>% select(feature, first, second,prob) %>% 
  arrange(desc(prob)) 
preds[1:10,]


# building a 3 gram model
V = nrow(f1) #???

trigrambyrow <- function(row)
{ 
  words <- unlist(strsplit(row[1],"_"))
  freq<- as.numeric(row[2])
  bigrampair <- f2 %>% filter(first == words[1] & second == words[2])#bad bad coding - f2 is accessed from global scope.
  prewordfreq <- as.numeric(bigrampair$frequency)
  return(freq/( prewordfreq + V))
}

ts <- Sys.time()
trigram_probs <- apply(f3, 1, trigrambyrow)
te <- Sys.time()
time_elapsed = te- ts;
print("Time taken to calculate the probs");print(time_elapsed) #21 mins!

f3$first <- apply(f3,1,function(row){w <- unlist(strsplit(row[1],"_"))[1]})
f3$second <- apply(f3, 1, function(row){w <-unlist(strsplit(row[1],"_"))[2]})
f3$third <- apply(f3, 1, function(row){w <-unlist(strsplit(row[1],"_"))[3]})
f3$probs <- as.numeric(trigram_probs)

#write the f2 and f3 to files.
write.csv2(f2,"bigrams.csv")
write.csv2(f3,"trigrams.csv")

# prediction
word1 <- "one"; word2 <- "of"
preds <- f3 %>% filter(first == word1 & second == word2) %>% 
  select(feature, first, second,third,probs) %>% 
  arrange(desc(probs)) 
preds[1:10,]


#*************** Verification and validation *********************************

set.seed(12345)
toks <- quanteda::tokens(TestData[1:1000,], what = "word")
toks_bigram <- tokens_ngrams(toks, n= 2)
toks_3gram <- tokens_ngrams(toks, n= 3)

m2 <- dfm(toks_bigram)
testf2 <- textstat_frequency(m2)[1:5000,]
m3 <- dfm(toks_3gram)
testf3 <- textstat_frequency(m3)[1:5000,]


# Calculating the Accuracy at first second and third predictions for bigram model

# for each bigram input first word and get first three predictions. count the accuracy.
getBigram_match<- function(row){
  input <- unlist(strsplit(row[1],"_"))
  #print("input");print(input)
  
  preds <- f2 %>% filter(first == input[1]) %>% select(feature, first, second,prob) %>% 
    arrange(desc(prob)) 
  matches <- preds[1:3,]$second
  
  match1 =0
  if(nrow(preds) >0){match1 = ifelse(input[2]== matches[1],1,0)}
  match2 = 0
  if(nrow(preds) >1){match2 = ifelse(input[2]== matches[2],1,0)}
  match3 = 0
  if(nrow(preds) >2){match3 = ifelse(input[2]== matches[3],1,0)}
  #print(preds[1:5,])
  res <- c(input[1],
           input[2],
           match1,
           match2,
           match3,matches[1],matches[2],matches[3])
  #print(res)
  return(res)
}

start_time <- Sys.time()
matchmatrix <- t(apply(testf2,1,getBigram_match))
end_time <- Sys.time()
time_elapsed <- end_time -start_time
print(paste("time elapsed",time_elapsed))

matchmatrix <- as.data.frame(matchmatrix)
names(matchmatrix) <- c("word1","word2","m1","m2","m3","p1","p2","p3")
n <- nrow(matchmatrix)

AccuracyFirstWord <- sum(as.numeric(levels(matchmatrix$m1))[matchmatrix$m1])/n
AccuracySecondWord <- sum(as.numeric(levels(matchmatrix$m2))[matchmatrix$m2])/n
AccuracyThirdWord <- sum(as.numeric(levels(matchmatrix$m3))[matchmatrix$m3])/n 

# Calculating accuracy at first three predictions for trigram

getTrigram_match<- function(row){
  input <- unlist(strsplit(row[1],"_"))
  preds <- f3 %>% filter(first == input[1] & second == input[2]) %>% 
    select(feature, first, second,third,probs) %>%  arrange(desc(probs))
  #print(preds[1:3,])
  matches <- preds[1:3,]$third
  match1 =0
  if(nrow(preds) >0){match1 = ifelse(input[3]== matches[1],1,0)}
  match2 = 0
  if(nrow(preds) >1){match2 = ifelse(input[3]== matches[2],1,0)}
  match3 = 0
  if(nrow(preds) >2){match3 = ifelse(input[3]== matches[3],1,0)}
  res <- c(input[1],input[2],input[3],match1, match2,match3,
           matches[1],matches[2],matches[3])
  return(res)
}

start_time <- Sys.time()
matchmatrix2 <- t(apply(testf3,1,getTrigram_match))
end_time <- Sys.time()
time_elapsed <- end_time -start_time
print(paste("time elapsed",time_elapsed))

matchmatrix2 <- as.data.frame(matchmatrix2)
names(matchmatrix2) <- c("word1","word2","word3","m1","m2","m3","p1","p2","p3")
n <- nrow(matchmatrix2)

AccuracyFirstWord_tri <- sum(as.numeric(levels(matchmatrix2$m1))[matchmatrix2$m1])/n
AccuracySecondWord_tri <- sum(as.numeric(levels(matchmatrix2$m2))[matchmatrix2$m2])/n
AccuracyThirdWord_tri <- sum(as.numeric(levels(matchmatrix2$m3))[matchmatrix2$m3])/n 

######## for quiz

word1 = "settle";word2 = "the"

f2 %>% filter(first == word2) %>% select(feature, first, second,prob) %>% 
  arrange(desc(prob))

f3 %>% filter(first == word1 & second == word2) %>% 
  select(feature, first, second,third,probs) %>%  arrange(desc(probs))


#remove objects
#***********************************************************************************









