################ imports used ################

library(Rtsne)
library(ggplot2)
library(caret)
library(tidyverse)
library(data.table)
library(xgboost)

################ read in data ################

master <- fread("./project/volume/data/interim/master.csv")

################ run pca ################

# pull out id, category cols
non_pcas <- select(master, id, category)
master = subset(master, select = -c(id, category))

# perform pca
pca <- prcomp(master)

# pca screeplot
screeplot(pca)

# use unclass() to get data in PCA space
pca_dt <- data.table(unclass(pca)$x)

# run t-snes on PCAs
tsne <- Rtsne(pca_dt, pca = F, perplexity = 75, check_duplicates = F)

# grab out coords from t-sne
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x = V1, y = V2)) + geom_point()

################ fit XGBoost model to tsne output ################

# separate train, test values from tsne
train_tsne <- head(tsne_dt, 250)
test_tsne <- tail(tsne_dt, 24750)

# add non_pca columns back to train, test
train_non_pcas <- head(non_pcas, 250)
train_tsne <- cbind(train_non_pcas, train_tsne)

test_non_pcas <- tail(non_pcas, 24750)
test_tsne <- cbind(test_non_pcas, test_tsne)

### subset out id columns, and store response vars
drops <- c('id')

train_tsne <- train_tsne[, !drops, with = F]
test_tsne  <- test_tsne[, !drops, with = F]

y.train_tsne <- train_tsne$category
y.test_tsne <- test_tsne$category

dummies <- dummyVars(category ~ ., data = train_tsne)

x.train_tsne <- predict(dummies, newdata = train_tsne)
x.test_tsne <- predict(dummies, newdata = test_tsne)

dtrain_tsne <- xgb.DMatrix(x.train_tsne, label = y.train_tsne, missing = NA)
# dtest_tsne <- xgb.DMatrix(x.test_tsne, missing = NA)

################ cross validation ################

hyper_perm_tune <- NULL

param <- list( objective = "multi:softprob",
               eval_metric = "mlogloss",
               num_class = 11
)


XGBm <- xgb.cv( params = param, nfold = 5, nrounds = 10000, missing = NA, data = dtrain_tsne, print_every_n = 1, 
                early_stopping_rounds = 25)

best_ntrees <- unclass(XGBm)$best_iteration

new_row <- data.table(t(param))     
new_row$best_ntrees <- best_ntrees

test_error <- unclass(XGBm)$evaluation_log[best_ntrees,]$test_mlogloss_mean
new_row$test_error <- test_error

hyper_perm_tune <- rbind(new_row, hyper_perm_tune)

################ fit a model ################

watchlist <- list(train = dtrain_tsne)

XGBm <- xgb.train( params = param, nrounds = best_ntrees, missing = NA, data = dtrain_tsne, watchlist = watchlist, print_every_n = 1)

################ save model and dummies ################

saveRDS(XGBm, "./project/volume/models/xgb_model.model")
saveRDS(dummies, "./project/volume/models/xgb_dummies.dummies")

################ save tsne test output ################
fwrite(x.test_tsne, './project/volume/data/processed/test_tsne.csv')
