import numpy as np

def long_tail_skew(num_class, ratio=1.0):
    alpha = np.log(ratio) / (num_class - 1)
    t = np.arange(num_class)
    decay_array = np.exp(-alpha * t)

    # print(decay_array)
    # [1.0, ..., 1/ratio]
    total = np.sum(decay_array)
    decay_array /= total

    return decay_array


def group_non_iid(num_group, num_ingroup_clients, num_classes, bias):
    if bias == 0:
        label_distribution = np.full(
            (num_classes, num_group*num_ingroup_clients), 
            1.0 / num_group/ num_ingroup_clients
        )
    else:
        alpha = 1.0 / bias
        group_label_distribution = np.random.dirichlet(
            alpha * np.ones(num_group), 
            size=num_classes
        ) / num_ingroup_clients

        label_distribution = np.tile(group_label_distribution, (1, num_ingroup_clients))

    # print(label_distribution)
    return label_distribution


def load_clients(batch_num_per_client=2000, num_class=50, num_group=4, num_ingroup_clients=1, ratio=1.0, bias=1):
    batch_num_per_client = num_class * 10 * num_group * num_ingroup_clients
    long_tail_distribution = long_tail_skew(num_class, ratio)
    non_iid_distribution = group_non_iid(num_group, num_ingroup_clients, num_class, bias).T

    batch_distribution = np.array([
        np.array([
            x for x in (long_tail_distribution * non_iid_distribution[idx] * batch_num_per_client)
        ]) for idx in range(num_group*num_ingroup_clients)
    ])

    batch_distribution = np.floor(batch_distribution).astype(int)

    # print(batch_distribution)
    # for row in batch_distribution:
    #     print(sum(row))

    return batch_distribution



if __name__ == "__main__":
    from .load_data import *

    def test(batch_distribution):
        test_loader = Ucf101Dataset(
            image_dir_list_file = "/data0/wyliang/datasets/ucf101/ucfTrainTestlist/testlist01.txt", 
            img_dir_root="/data0/wyliang/datasets/ucf101/UCF-101-frames/", 
            image_size=224, 
            mode="test", 
            shuffle=False, 
            class_distribution=batch_distribution,
            step=20
        )

    batch_distribution = load_clients()
    test(batch_distribution[0])