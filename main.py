import cv2
import numpy as np
import additional_functions as src
from os import listdir

fonts = ['Avenir', 'Courier', 'Marion']
path_to_train_samples = 'assets/images/train/'
path_to_output_vectors = 'output/vectors/'

print('\n\n\nHello!')

# [===>--<==>--<==>--<==>--<==>    start of settings    <==>--<==>--<==>--<==>--<===]

# chose one of the fonts from 'fonts' array
font = fonts[0]

# set background color in R,G,B style, where R,G and B are strings of 2 chars
background_color = ['FF', 'FF', 'FF']

# set background color range parameter
range_param = 15

# is target image is darker than background?
target_is_darker = True

# [===>--<==>--<==>--<==>--<==>     end of settings     <==>--<==>--<==>--<==>--<===]

path_to_train_samples += font + '/'

train_samples_list = listdir(path_to_train_samples)
train_samples_list.remove('.DS_Store')

vector_length = 0

for index in range(0, len(train_samples_list)):
    img_vector = src.image_to_array(path_to_train_samples + train_samples_list[index],
                                    background_color,
                                    range_param,
                                    target_is_darker)

    filename = 'output/vectors/vector-' + train_samples_list[index][0]
    np.save(filename, img_vector)

    if vector_length == 0:
        vector_length = img_vector.shape[1]
    if img_vector.shape[1] != vector_length:
        print 'WARNING: different length of vectors detected'

print 'vectors has built successfully'
print 'length of vectors is ' + str(vector_length) + ' chars\n'

vector_list = listdir(path_to_output_vectors)

N = len(vector_list)

print('Start building of W matrix...')
W = [[0] * vector_length for _ in range(vector_length)]
print('W matrix has built.\n\nTraining has begun...')
for i in range(0, N):
    print('BEGIN STAGE ' + str(i+1) + '/' + str(N))
    x = np.load(path_to_output_vectors + vector_list[i])
    print 'stage ' + str(i+1) + '/' + str(N) + ': array was loaded'
    # xT = x.T
    # print 'stage ' + str(i+1) + '/' + str(N) + ': array was transposed'
    w = np.outer(x, x)
    print 'stage ' + str(i+1) + '/' + str(N) + ': arrays was multiplied'
    W = np.add(W, w)
    print('stage ' + str(i+1) + '/' + str(N) + ': sum successful')
    print 'COMPLETE STAGE ' + str(i+1) + '/' + str(N) + '\n'

W = np.dot(W, 1/N)

print W

for i in range(vector_length):
    for j in range(vector_length):
        if W[i][j] != 0:
            print 'stageB'
            print W[i][j]
            print '---------------'
print '\n\n\n'
