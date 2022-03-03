import os


def generate(dir,label):
    files = os.listdir(dir)
    files.sort()
    print
    '****************'
    print
    'input :', dir
    print
    'start...'
    listText = open('amazon_reorgnized.txt', 'a')  # generated file's name
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = dir +  '/' + file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()
    print
    'down!'
    print
    '****************'


outer_path = 'root/path/to/your/datasets' # Put the pictures in different folders according to their class, the folder is named the class name, make sure to sort alphabetically.

if __name__ == '__main__':
    i = 0  # The start class ID, generating i-th --> [i + 'the number of your folders in the outer_path']-th classes
    folderlist = os.listdir(outer_path)
    folderlist.sort()
    for folder in folderlist:
        generate(os.path.join(outer_path, folder),i)
        i += 1
