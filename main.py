import numpy as np
import additional_functions as src
from os import listdir

train_sets = ['digits', 'Avenir', 'Courier', 'Marion']
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
max_iterations = 15

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
# initialize_vectors()

vector_list = listdir(path_to_output_vectors)

n_elements = np.load(path_to_output_vectors + vector_list[0]).shape[1]
print(n_elements)
n_vectors = len(vector_list)


def hamming_distance(a, b):
    c = 0
    for i in range(n_elements):
        c += a[0, i] * b[0, i]
    return (n_elements - c) / 2


def train_network():
    print('Start building of W matrix...')
    weights_matrix = np.matrix([[0] * n_elements for _ in range(n_elements)])
    print('W matrix has built.\n\nTraining has begun...\n')

    for i in range(0, n_vectors):
        print('Begin stage ' + str(i + 1) + '/' + str(n_vectors) + '')
        x = np.load(path_to_output_vectors + vector_list[i])
        weights_matrix = weights_matrix + np.outer(x, x)
        print('Stage complete ' + str(i + 1) + '/' + str(n_vectors) + '\n')

    weights_matrix = weights_matrix / n_vectors

    for i in range(0, n_elements):
        weights_matrix[i, i] = 0

    print(weights_matrix)
    print(weights_matrix.shape)

    np.save('output/matrix/matrix', weights_matrix)
    print('TRAINING COMPLETE\n\n\n')


train_network()
W = np.load('output/matrix/matrix.npy')
np.savetxt('output/matrix/matrix.txt', W, delimiter=', ', fmt='%s')
print ('Weights matrix has loaded')

v = src.image_to_array('input/7.bmp',
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
    y = v.copy()
    y_next = W * y.copy()
    for i in range(y.shape[1]):
        y_next[0, i] = threshold(y_next[0, i])
    distance = hamming_distance(y, y_next)
    print(distance)
    is_result_found = distance == 0
    if k == max_iterations | is_result_found:
        break

if is_result_found:
    src.cv2.imshow('123', np.reshape(y, (image_shape[0], image_shape[1])))
else:
    print('no result')
