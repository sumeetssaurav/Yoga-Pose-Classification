import os


f1 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/trainlist03.txt','r')
f2 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/testlist03.txt','r')

train_list = f1.readlines()
test_list = f2.readlines()

f3 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/train_file03.txt', 'w')
f4 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/test_file03.txt', 'w')



for line in train_list:
    name = line.split(' ')[0]
    label = line.split(' ')[-1]
    print(label)
    f3.write(name[:-4]+' '+ str(int(label)-1)+'\n')


for line in test_list:
    name = line.split(' ')[0]
    label = line.split(' ')[-1]
    f4.write(name[:-4]+' '+str(int(label)-1)+'\n')

f1.close()
f2.close()
f3.close()
f4.close()
