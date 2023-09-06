import os

img_path = '/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/Images/'
f1 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/train_file03.txt','r')
f2 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/test_file03.txt','r')

train_list = f1.readlines()
test_list = f2.readlines()

f3 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/train_list03.txt', 'w')
f4 = open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/test_list03.txt', 'w')

clip_length = 64

for line in train_list:
    name = line.split(' ')[0]
    print(name)
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // clip_length
    for i in range(nb):
        f3.write(name+' '+ str(i*clip_length+1)+' '+label)


for line in test_list:
    name = line.split(' ')[0]
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // clip_length
    for i in range(nb):
        f4.write(name+' '+ str(i*clip_length+1)+' '+label)

f1.close()
f2.close()
f3.close()
f4.close()
