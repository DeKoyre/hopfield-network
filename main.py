import numpy as np
import additional_functions as src
from os import listdir

train_sets = ['icons', 'digits', 'Avenir', 'Courier', 'Marion']
path_to_train_samples = 'assets/images/train/'
path_to_output_vectors = 'output/vectors/'

print('\n\n\nHello!')

# [===>--<==>--<==>--<==>--<==>    start of settings    <==>--<==>--<==>--<==>--<===]

# chose one of the sets from 'train_sets' array
train_set = train_sets[0]

# set background color in R,G,B style, where R,G and B are strings of 2 chars
background_color = ['FF', 'FF', 'FF']

# set background color range parameter
range_param = 15

# is target image is darker than background?
target_is_darker = True

# is target image is darker than background?
max_iterations = 20

# is init
is_init = False

# [===>--<==>--<==>--<==>--<==>     end of settings     <==>--<==>--<==>--<==>--<===]
#
#
#
#
#
#
#
# [===>--<==>--<==>--<==>--<==>   start of main part    <==>--<==>--<==>--<==>--<===]

path_to_train_samples += train_set + '/'

train_samples_list = listdir(path_to_train_samples)
if src.isfile(path_to_train_samples + '.DS_Store'):
    train_samples_list.remove('.DS_Store')


def initialize_vectors():
    number_of_chars = 0

    for index in range(0, len(train_samples_list)):
        img_vector = src.image_to_array(path_to_train_samples + train_samples_list[index],
                                        background_color,
                                        range_param,
                                        target_is_darker)

        filename = 'output/vectors/vector-' + train_samples_list[index][0]
        np.save(filename, img_vector)

        if number_of_chars == 0:
            number_of_chars = img_vector.shape[1]
        if img_vector.shape[1] != number_of_chars:
            print('WARNING: different length of vectors detected')

    print('Vectors has built successfully')
    print('Length of each vector is ' + str(number_of_chars) + ' chars\n')


img = src.cv2.imread(path_to_train_samples + train_samples_list[0], src.cv2.IMREAD_COLOR)
image_shape = img.shape
if is_init:
    initialize_vectors()

vector_list = listdir(path_to_output_vectors)

n_elements = np.load(path_to_output_vectors + vector_list[0]).shape[1]
print(n_elements)
n_vectors = len(vector_list)


def hamming_distance(a, b):
    c = 0
    for i in range(n_elements):
        c += a[i, 0] * b[i, 0]
    return (n_elements - c) / 2


def train_network():
    print('Start building of W matrix...')
    weights_matrix = np.matrix([[0] * n_elements for _ in range(n_elements)])
    print('W matrix has built.\n\nTraining has begun...\n')

    for i in range(0, n_vectors):
        print('Begin stage ' + str(i + 1) + '/' + str(n_vectors) + '')
        x = np.load(path_to_output_vectors + vector_list[i])
        weights_matrix += np.outer(x, x)
        print('Stage complete ' + str(i + 1) + '/' + str(n_vectors) + '\n')

    # print(type(weights_matrix))
    # print(weights_matrix)
    # print(weights_matrix / 4)

    for i in range(0, n_elements):
        weights_matrix[i, i] = 0

    arrays = []
    for d in range(n_vectors):
        arrays.append(np.load(path_to_output_vectors + vector_list[d]))

    for i in range(n_elements):
        for j in range(n_elements):
            for d in range(n_vectors):
                weights_matrix[i, j] += arrays[d][0, i] * arrays[d][0, j]

    weights_matrix = np.matrix(weights_matrix / n_elements)

    np.save('output/matrix/matrix', weights_matrix)
    print('TRAINING COMPLETE\n\n\n')


if is_init:
    train_network()

W = np.matrix(np.load('output/matrix/matrix.npy'))
np.savetxt('output/matrix/matrix.txt', W, delimiter=', ', fmt='%s')
print ('Weights matrix has loaded')

# x = src.image_to_array('input/7.bmp')
# # print (hamming_distance(x, (W * x.T).T))
# print(np.equal(x, (W * x.T).T))
# # print(((W * x.T).T).shape)

v = src.image_to_array('input/1.bmp',
                       background_color,
                       range_param,
                       target_is_darker)


def threshold(a):
    if a >= 0:
        return 1
    else:
        return -1


k = 0

while True:
    k += 1
    y = v.T.copy()
    y_next = W * y.copy()
    for i in range(n_elements):
        y_next[i, 0] = threshold(y_next[i, 0])
    distance = hamming_distance(y, y_next)
    is_result_found = distance == 0
    if k == max_iterations | is_result_found:
        break

if is_result_found:
    print ('result')
    # src.cv2.imshow('123', np.reshape(y, (image_shape[0], image_shape[1])))
else:
    print('no result')
