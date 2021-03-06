import numpy as np
import additional_functions as src
from os import listdir

train_sets = ['icons', 'digits', 'Avenir', 'Courier', 'Marion']
path_to_train_samples = 'assets/images/train/'
path_to_output_vectors = 'output/vectors/'

print('\n\n\nHello!')

# [===>--<==>--<==>--<==>--<==>    start of settings    <==>--<==>--<==>--<==>--<===]

# chose one of the sets from 'train_sets' array
train_set = train_sets[1]

# set background color in R,G,B style, where R,G and B are strings of 2 chars
background_color = ['FF', 'FF', 'FF']

# set background color range parameter
range_param = 15

# is target image is darker than background?
target_is_darker = True

# is target image is darker than background?
max_iterations = 20

# is init
is_init = True

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

    for i in range(0, len(train_samples_list)):
        img_vector = src.image_to_array(path_to_train_samples + train_samples_list[i],
                                        background_color,
                                        range_param,
                                        target_is_darker)

        filename = 'output/vectors/vector-' + train_samples_list[i][0]
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

v = src.image_to_array('input/' + train_set + '/x.bmp',
                       background_color,
                       range_param,
                       target_is_darker)


def threshold(a):
    if a >= 0:
        return 1
    else:
        return -1


k = 0
y = v.T.copy()
is_result_found = False
print('input loaded. Use network')
while k < max_iterations and not is_result_found:
    k += 1
    y_next = W * y.copy()
    for index in range(n_elements):
        y_next[index, 0] = threshold(y_next[index, 0])
    distance = hamming_distance(y, y_next)
    print(str(k) + ': ' + str(distance))
    is_result_found = distance == 0
    y = y_next

if is_result_found:
    print('result')
    result = y.copy()
    np.place(result, result == -1.0, 255)
    np.place(result, result == 1.0, 0)

    src.cv2.imshow('123', np.reshape(result, (image_shape[0], image_shape[1])))
    src.cv2.waitKey()
else:
    print('no result')
