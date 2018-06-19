
#join the train data
bash join.sh train 7 querytid_tokensid.txt 1 > train1
bash join.sh train1 8 purchaseid_tikensid.txt 1 > train2
bash join.sh train2 9 titleid_tokensid.txt 1 > train3
bash join.sh train3 10 descriptionid_tokensid.txt 1 > train4
bash join.sh train4 11 userid_profile.txt 1 > train5

#after the join operation,the column name is the bellow order:
#keep in mind, some line doesn't has "geder" and "age" because the "userID" is 0 represents "unknow"
#[urlID	adID	advertiserID depth	queryID	keywordID	titleID	descriptionID	
# userID position queryTokens	keywordsTokens	titleTokens	descriptionTokens	gender	age]

#before join the test data, we first add line number in the column 1 to every line in order to get
#the same order after we perform "sort" and "join" operation
awk 'BEGIN{FS="\t"} {print FNR FS $0}' test > test_line_num

bash join.sh test_line_num 7 querytid_tokensid.txt 1 > test1
bash join.sh test4 8 purchaseid_tikensid.txt 1 > test2
bash join.sh test4 9 titleid_tokensid.txt 1 > test3
bash join.sh test4 10 descriptionid_tokensid.txt 1 > test4
bash join.sh test4 11 userid_profile.txt 1 > test5
#get the original order of test data according to the first column 
sort -t "\t" -k 1n test5 >test_resort

#extract feature from the train data and test data,in order to 
#get the same feature index, we use both trian and test data simultaneously
#as input data, after this operation we get train_feature and test_feature
python extract_feature.py train5 test_resort

#split train_feature to train_model_feature and validation_model_feature,we perform 
#adjusting-parameter in validation_model_feature
sort -R train_feature > train_shuffle_feature
head -n 500000 train_shuffle_feature > train_model_feature
tail -n 500000 train_shuffle_feature > validate_model_feature

#after we get the optimal parameters,we use this model to predict the test data
#we use "lr_ftrl.py"(follow the regularized leader_LR) to trian a model
python lr_ftrl.py train_feature test_feature

#we get the prediction of test data in "pctr_ftrl", the format like "[line_number,predict]
#we collect the predict use the follow bash:
awk 'BEGIN{FS=","}{print $2}' pctr_ftrl > predict.csv
