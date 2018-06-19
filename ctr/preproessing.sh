awk 'BEGIN{FS="\t"} {print FNR FS $0}' test > test_line_num

bash join.sh test_line_num 7 querytid_tokensid.txt 1 > test1
bash join.sh test4 8 purchaseid_tikensid.txt 1 > test2
bash join.sh test4 9 titleid_tokensid.txt 1 > test3
bash join.sh test4 10 descriptionid_tokensid.txt 1 > test4
bash join.sh test4 11 userid_profile.txt 1 > test5
sort 

