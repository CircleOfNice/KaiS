from math import sin, tan
import random

from algorithm_torch.GPG import get_piecewise_linear_fit_baseline
import matplotlib.pyplot as plt

length = 100
elems = [i*0.1 for i in range(length)]

func = lambda x: 2*x +2
straight_curve = [func(i) for i in range(length)]
sin_curve = [sin(elem) for elem in elems]
tan_curve = [tan(elem) for elem in elems]


def plot(x, y, title):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(x,y)
    plt.title(title)
    
    plt.savefig(title + '.png')
    plt.show()

plot(elems, straight_curve, 'straight_line')
plot(elems, sin_curve, 'sin_curve')
plot(elems, tan_curve, 'tan_curve')

r_indexes =list(random.sample(range(length), k=70))
r_indexes.sort(reverse=True)
print('remove_indexes : ', r_indexes)

def remove_indexes(list, r_indexes):
    for index in r_indexes:
        #print(index)
        del list[index]
    return list

elems_removed_index = remove_indexes(elems, r_indexes)

straight_curve_removed_index = remove_indexes(straight_curve, r_indexes)
plot(elems, straight_curve_removed_index, 'straight_line_with_index_removed')
straight_vals = get_piecewise_linear_fit_baseline([straight_curve_removed_index], [elems])
plot(elems, straight_vals[0], 'straight_line_removed_piece_wise_linear_fit')



sin_curve_removed_index = remove_indexes(sin_curve, r_indexes)
plot(elems, sin_curve_removed_index, 'sin_curve_with_index_removed')
sin_vals = get_piecewise_linear_fit_baseline([sin_curve_removed_index], [elems])
plot(elems, sin_vals[0], 'sin_curve_removed_piece_wise_linear_fit')



tan_curve_removed_index = remove_indexes(tan_curve, r_indexes)
plot(elems, tan_curve_removed_index, 'tan_curve_with_index_removed')
tan_vals = get_piecewise_linear_fit_baseline([tan_curve_removed_index], [elems])
plot(elems, tan_vals[0], 'tan_curve_removed_piece_wise_linear_fit')

def shuffle_appropirately(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
        
    return list1_shuf, list2_shuf


elems_1, sin_curve = shuffle_appropirately(elems.copy(), sin_curve)
elems_2, tan_curve = shuffle_appropirately(elems.copy(), tan_curve)



#sin_curve_removed_index = remove_indexes(sin_curve, r_indexes)
plot(elems_1, sin_curve, 'shuffled_sin_curve_with_index_removed')
sin_vals = get_piecewise_linear_fit_baseline([sin_curve], [elems])
plot(elems_1, sin_vals[0], 'shuffled_sin_curve_removed_piece_wise_linear_fit')



#tan_curve_removed_index = remove_indexes(tan_curve, r_indexes)
plot(elems_2, tan_curve, 'shuffled_tan_curve_with_index_removed')
tan_vals = get_piecewise_linear_fit_baseline([tan_curve], [elems])
plot(elems_2, tan_vals[0], 'shuffled_tan_curve_removed_piece_wise_linear_fit')