# New number to insert
import os
from collections import Counter
import time
'''
category = {
    'drone': 0,
    'phone': 1,
    'keyboard': 2,
    'shoe': 3
    }'''

category = {
    'drone': 0,
    'phone': 1,
    'keyboard': 2,
    'shoe': 3
    }

tasks = {
    'test': 0,
    'train': 1,
    'valid': 2
    }

file_direc = 'C:/Users/J Kit/Desktop/Y3 Intern/Week3-CollectRoboflowDataset/RoboflowDataset'
files = os.listdir(file_direc)

#17.16
test_ = []
train_ = []
valid_= []
for task in tasks:
    for cats in files:
        #if cats in category:
        if cats == 'phone':
            new_number = category[cats]
            print(new_number)
            dirs = file_direc + '/' + cats + '/' + task + '/labels'
            ans = os.listdir(str(dirs))
            num_files = len(ans)
            iterator = 0
            for file in range(len(ans)):
                file_path = os.path.join(dirs, ans[file])

                with open(file_path, 'r') as file:
                    content = file.readlines()
                
                test_.append(int(content[0][0]))

        
print(len(test_))
print(print(Counter(test_)))
print()
all_zero = all(x == 1 for x in test_)

print(all_zero)
time.sleep(5)
#file_path = '0001_jpg.rf.2a58e3cb39bf91af5e1c77b37400dce7.txt'




# Replace all occurrences of '0' in the first column with '3'

# Write the modified content back to the file



#print("First number replaced successfully.")
