#! /usr/bin


#python extract_feature.py train5 train_feature


#sort -R 1_feature > train_shuffle_feat
head -n 500000 1_feature_shuffle > train_model_feat
tail -n 500000 1_feature_shuffle > test_model_feat

