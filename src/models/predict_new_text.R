################ imports used ################

library(data.table)

################ read in test and models ################

xgb.model <- readRDS("./project/volume/models/xgb_model.model")
test_tsne <- fread('./project/volume/data/processed/test_tsne.csv')

################ fit model to test data ################

# transform test tsne output into matrix
mtest_tsne <- matrix(unlist(test_tsne), ncol = 2, nrow = 24750)
dtest_tsne <- xgb.DMatrix(mtest_tsne, missing = NA)

pred <- predict(xgb.model, newdata = dtest_tsne)

################### transform predictions and write out submission #######################

results <- matrix(pred, ncol = 11, byrow = T)
results <- data.table(results)

colnames(results) <- c('redditCars', 'redditCFB', 'redditCooking', 'redditMachineLearning', 'redditmagicTCG', 'redditpolitics', 
                       'redditRealEstate', 'redditscience', 'redditStockMarket', 'reddittravel', 'redditvideogames')
id <- 1:nrow(results)
submission <- cbind(id, results)
fwrite(submission, "./project/volume/data/processed/submission.csv")
