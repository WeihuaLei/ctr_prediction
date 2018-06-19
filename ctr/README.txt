#CTR prediction Project
#AUTHOR: LEIWEIHUA



#The source file and description

***process.sh***
all needed operation to perform ctr prediction, from raw data to
prediction results

***extract_feature.py***
python source file to extract feature from raw data

***lr_ftrl.py***
source file of train a LR model and perform the prediction

***tf.py***
utility of calculate tf-idf

***auc.py***
validate the model on validation data to adjust parameters

***join.sh***
join the train or test data to the auxiliary file like queryid_tokensid.txt

