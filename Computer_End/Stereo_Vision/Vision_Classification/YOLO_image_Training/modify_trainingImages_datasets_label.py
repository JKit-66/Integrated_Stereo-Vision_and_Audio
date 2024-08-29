# New number to insert
import os

category = {
    'drone': 0,
    'phone': 1,
    'keyboardk': 2,
    'shoek': 3
    }

'''
category = {
    'keyboardK': 2,
    'shoe': 3
    }
'''

tasks = {
    'test': 0,
    'train': 1,
    'valid': 2
    }

file_direc = 'C:/Users/J Kit/Desktop/Y3 Intern/Week3-CollectRoboflowDataset/RoboflowDataset'
files = os.listdir(file_direc)


for task in tasks:
    for cats in files:
        if cats in category:
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

                modified_content = [line.replace('0', str(new_number), 1) for line in content]

                with open(file_path, 'w') as file:
                    file.writelines(modified_content)
    
            #print(iterator, cats, file_path)
        print(f'{cats}-{task}- done')
        
print("First number replaced successfully.")
