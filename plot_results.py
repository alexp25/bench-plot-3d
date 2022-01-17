import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import copy
import numpy as np
from pprint import pprint
import json

mode = "test"
mode = "mp"
# mode = "opencv"
folder_index = 6
headers = None
interpolate = False
save_plot_flag = True
result_scale_seconds = False

client_col = "ClientSize"
delay_col = "DelaySize"

client_label = "number of clients"
delay_label = "request delay (ms)"

# client_col = "GroupSize"
# delay_col = "DelaySize"

# client_label = "number of groups"
# delay_label = "request delay (ms)"

if mode == "mp":
    folder = "./results/mp/" + str(folder_index)
    file = "bench_mp_results.json"
    output_img_3d = folder + "/" + "bench_mp_3d.png"
    output_img_cmap = folder + "/" + "bench_mp_cmap.png"
    result_scale_seconds = True
elif mode == "opencv":
    folder = "./results/media/" + str(folder_index)
    file = "bench_opencv_results.txt"
    output_img_3d = folder + "/" + "bench_opencv_3d.png"
    output_img_cmap = folder + "/" + "bench_opencv_cmap.png"
    result_scale_seconds = True
else:
    folder = "."
    file = "test_results.txt"
    output_img_3d = folder + "/" + "bench_test_3d.png"
    output_img_cmap = folder + "/" + "bench_test_cmap.png"

print(folder + "/" + file)

with open(folder + "/" + file, 'r') as csvfile:
    data1 = json.load(csvfile)
    data = data1["Data"]["squares"]
    headers_settings = data1["Data"]["settings"]

# pprint(data)

# print(np.append(np.array([1,2]), 3))


def get_interpolated_array(values):
    n = len(values)
    x_axis = range(n)
    # print(x_axis)
    interpolated_values = np.interp(np.linspace(np.min(x_axis), np.max(x_axis), num=10*n), x_axis, values)
    return interpolated_values


def post_processing(data_matrix):
    for row in data_matrix:
        avg = np.average(row)
        # print(avg)
        row[row > avg] = avg
    return data_matrix

# d = np.array([1, 2, 3])
# print(get_interpolated_array(d))

def load_data_3d(data, interpolate):
    X = np.arange(1, 10)
    Y = np.arange(1, 10)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # print("X: ", X, "Y: ", Y, "Z: ", Z)

    n_col = len(data)
    n_row = len(data[0])
    print(n_col, n_row)

    x_all = np.zeros(shape=(n_col, n_row))
    y_all = np.zeros(shape=(n_col, n_row))
    z_all = np.zeros(shape=(n_col, n_row))

    for (i, col) in enumerate(data):
        for (j, row) in enumerate(col):
            n_clients = row['settings'][client_col]
            delay = row['settings'][delay_col]
            result = row['TimeResult']
            x_all[i][j] = n_clients
            y_all[i][j] = delay
            z_all[i][j] = result

    # z_all = post_processing(z_all)
    # x_all = np.transpose(x_all)
    # y_all = np.transpose(y_all)

    # y_all = np.flip(y_all, 1)

    # flipped because the delay (x_axis) is reversed on the plot
    y_all = np.fliplr(y_all)
    # z_all = np.fliplr(z_all)

    if result_scale_seconds:
        z_all = z_all / 1000

    if interpolate:
        X = get_interpolated_array(X)
        Y = get_interpolated_array(Y)

    X = x_all
    Y = y_all
    Z = z_all
    print(np.shape(X), np.shape(Y), np.shape(Z))

    # print("X: ", X, "Y: ", Y, "Z: ", Z)
    return X, Y, Z


X, Y, Z = load_data_3d(data, interpolate)

def plot3d(X, Y, Z, interpolate):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Y = np.fliplr(Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_ylabel(delay_label)
    ax.set_xlabel(client_label)
    if result_scale_seconds:
        ax.set_zlabel('response time (s)')
    else:
        ax.set_zlabel('response time (ms)')

    # plt.axis('off')
    # Set rotation angle
    ax.view_init(azim=-140, elev=32)

    # ax.view_init(azim=180, elev=90)

    # ax.axes.xaxis.set_ticks([])
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_ticks([])
    # ax.axes.zaxis.set_ticks([])


    # plt.gca().axes.get_yaxis().set_visible(False)

    save_plot(plt, output_img_3d)
    # plt.show()

    return plt

def save_plot(plt, pname):
    global save_plot_flag
    if save_plot_flag:
        plt.savefig(pname, format='png', dpi=1200)


def plot_cmap(X, Y, Z, interpolate):
    # not showing correctly
    # ax = plt.imshow(Z, cmap='hot')
    fig, ax = plt.subplots()
    im = ax.imshow(np.transpose(Z), cmap='viridis')
    plt.colorbar(im, orientation='horizontal')
    xh, yh = get_headers_from_data(X, Y, Z, interpolate)

    # yh = np.flip(yh)

    # aux = yh
    # yh = xh
    # xh = aux

    ax.set_xticks(np.arange(len(xh)))
    ax.set_yticks(np.arange(len(yh)))
    ax.set_xticklabels(xh)
    ax.set_yticklabels(yh)

    save_plot(plt, output_img_cmap)

    # plt.show()

    return plt
    # fig, ax = plt.subplots()
    # im = ax.imshow(Z)
    #
    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(X)))
    # ax.set_yticks(np.arange(len(Y)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(X)
    # ax.set_yticklabels(Y)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(X)):
    #     for j in range(len(Y)):
    #         text = ax.text(j, i, Z[i, j],
    #                        ha="center", va="center", color="w")
    #
    # ax.set_title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()
    # plt.show()

def get_headers_from_data(X, Y, Z, interpolate):
    headers_x = np.unique(X)
    print(headers_x)
    headers_y = np.flip(np.unique(Y))
    headers_x = headers_x.astype(int)
    headers_y = headers_y.astype(int)
    return headers_x, headers_y


print(get_headers_from_data(X, Y, Z, interpolate))


f = plot_cmap(X, Y, Z, interpolate)
# g = plot_cmap(X, Y, Z, interpolate)

# f = plot3d(X, Y, Z, interpolate)
g = plot3d(X, Y, Z, interpolate)

f.show()
g.show()


