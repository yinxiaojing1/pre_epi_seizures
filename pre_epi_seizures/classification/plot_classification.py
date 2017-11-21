import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def get_label_list(data, labels, colors):
    colors = data[colors]
    labels = data[labels]

    return labels.unique(), colors.unique()


def jointplot(data, x, y, labels, colors):
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    label_list, color_list = get_label_list(data, labels, colors)


    label = 1
    color = 'g'
    data_to_plot = data.loc[data[labels] == label]
    data_to_plot_x = data_to_plot[x]
    data_to_plot_y = data_to_plot[y]
    print data_to_plot_x
    ax1.scatter(data_to_plot_x, data_to_plot_y, color=color, s=10)


    # label = 2
    # color = 'r'
    # data_to_plot = data.loc[data[labels] == label]
    # data_to_plot_x = data_to_plot[x]
    # data_to_plot_y = data_to_plot[y]
    # print data_to_plot_x
    # ax1.scatter(data_to_plot_x, data_to_plot_y, color=color, s=10)




    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax1.legend(label_list)

    print 'done'

    f1.savefig('joint')