import numpy as np
import calculate_wavelet
import scipy.io
import time

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 5
size_non_overlap = 5


def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    idx = 0
    seg = 0
    while idx + number_of_vector_per_example < len(vector_to_format):
        example = vector_to_format[idx:idx + number_of_vector_per_example]
        example = example.transpose()
        dataset_example_formatted.append(example)
        idx += size_non_overlap
        seg += 1
    t = time.time()
    data_calculated = calculate_wavelet.calculate_wavelet_dataset(dataset_example_formatted)
    return np.array(data_calculated), time.time() - t, seg


def read_data(path, type):
    print("Reading Data")
    list_dataset = []
    list_labels = []
    t = 0
    seg = 0
    if type == 'pre-training':

        for ty in range(2):
            for candidate in range(1, 31):
                labels = []
                examples = []
                for i in range(1, 6):
                    for rp in range(1, 25 + 1):
                        aux = scipy.io.loadmat(path + 'emg_person' + str(candidate) + '_class' + str(i) + '_rpt' + str(
                            rp) + '_type' + str(ty) + '.mat')
                        dataset_example, auxt, auxseg = format_data_to_train(aux['emg'])
                        t += auxt
                        seg += auxseg
                        examples.append(dataset_example)
                        labels.append(((i - 1) % number_of_classes) + np.zeros(dataset_example.shape[0]))

                list_dataset.append(examples)
                list_labels.append(labels)
                print(candidate)
                print('time', t / seg)
                t = 0
                seg = 0

    elif type == 'training':

        for candidate in range(31, 61):
            labels = []
            examples = []
            for i in range(1, 6):
                for rp in range(1, 25 + 1):
                    aux = scipy.io.loadmat(path + 'emg_person' + str(candidate) + '_class' + str(
                        i) + '_rpt' + str(rp) + '_type' + str(0) + '.mat')
                    dataset_example, auxt, auxseg = format_data_to_train(aux['emg'])
                    t += auxt
                    seg += auxseg
                    examples.append(dataset_example)
                    labels.append(((i - 1) % number_of_classes) + np.zeros(dataset_example.shape[0]))

            list_dataset.append(examples)
            list_labels.append(labels)
            print(candidate)
            print('time', t / seg)
            t = 0
            seg = 0
    elif type == 'testing':

        for candidate in range(31, 61):
            labels = []
            examples = []
            for i in range(1, 6):
                for rp in range(1, 25 + 1):
                    aux = scipy.io.loadmat(path + 'emg_person' + str(candidate) + '_class' + str(
                        i) + '_rpt' + str(rp) + '_type' + str(1) + '.mat')
                    dataset_example, auxt, auxseg = format_data_to_train(aux['emg'])
                    t += auxt
                    seg += auxseg
                    examples.append(dataset_example)
                    labels.append(((i - 1) % number_of_classes) + np.zeros(dataset_example.shape[0]))

            list_dataset.append(examples)
            list_labels.append(labels)
            print(candidate)
            print('time', t / seg)
            t = 0
            seg = 0

    print("Finished Reading Data")
    return list_dataset, list_labels
