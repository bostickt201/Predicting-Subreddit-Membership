################ imports used ################
library(tidyverse)
library(data.table)


################ read in data ################

train <- fread('./project/volume/data/raw/training_data.csv')
train_emb <- fread('./project/volume/data/raw/training_emb.csv')

test <- fread('./project/volume/data/raw/test_file.csv')
test_emb <- fread('./project/volume/data/raw/test_emb.csv')

################ create master set ################

# initialize 2 empty data frames
df_train <- data.frame(matrix(0, nrow = 250, ncol = 11))
df_test <- data.frame(matrix(0, nrow = 24750, ncol = 11))

names(df_train) <- c('redditCars', 'redditCFB', 'redditCooking', 
                     'redditMachineLearning', 'redditmagTCG',
                     'redditpolitics', 'redditRealEstate', 'redditscience',
                     'redditStockMarket', 'reddittravel', 'redditvideogames')

names(df_test) <- c('redditCars', 'redditCFB', 'redditCooking', 
                    'redditMachineLearning', 'redditmagTCG',
                    'redditpolitics', 'redditRealEstate', 'redditscience',
                    'redditStockMarket', 'reddittravel', 'redditvideogames')

# cbind initialize frames with train, test
train <- cbind(train, df_train)
test <- cbind(test, df_test)

# reformat train; subset reddit column
train <- within(train, redditCars[reddit == 'cars'] <- 1)
train <- within(train, redditCFB[reddit == 'CFB'] <- 1)
train <- within(train, redditCooking[reddit == 'Cooking'] <- 1)
train <- within(train, redditMachineLearning[reddit == 'MachineLearning'] <- 1)
train <- within(train, redditmagTCG[reddit == 'magicTCG'] <- 1)
train <- within(train, redditpolitics[reddit == 'politics'] <- 1)
train <- within(train, redditRealEstate[reddit == 'RealEstate'] <- 1)
train <- within(train, redditscience[reddit == 'science'] <- 1)
train <- within(train, redditStockMarket[reddit == 'StockMarket'] <- 1)
train <- within(train, reddittravel[reddit == 'travel'] <- 1)
train <- within(train, redditvideogames[reddit == 'videogames'] <- 1)

train <- train[, 2:13]

# add id, order columns to train and subset out text column
id <- 1:nrow(train)
order <- 1:nrow(train)
train <- cbind(id, order, train)

text_col_train <- select(train, text, order)
train = subset(train, select = -c(text))

m_train <- melt(train, id = c('id', 'order'), variable.name = 'category')
m_train <- m_train[value == 1][order(order)][,.(id, category)]
m_train$category <- as.integer(m_train$category) - 1

# reformat test so it looks like m_train
order <- 1:nrow(test)
test <- cbind(order = order, test)

text_col_test <- select(test, text, order)
test = subset(test, select = -c(text))

test2 <- data.table(test$id)
test2$category <- NA
names(test2)[names(test2) == 'V1'] <- 'id'

# cbind embeddings to test, train data
m_train <- cbind(m_train, train_emb)
test2 <- cbind(test2, test_emb)

# finish master by rbinding train, test
master <- rbind(m_train, test2)

# write out and save master under processed data
fwrite(master, "./project/volume/data/interim/master.csv")
