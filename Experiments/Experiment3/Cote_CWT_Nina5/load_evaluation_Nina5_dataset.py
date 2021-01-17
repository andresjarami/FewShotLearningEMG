import scipy.io as sio
import numpy as np
import calculate_wavelet
import os

def format_data_to_train_cwt(vector_to_format, number_of_vector_per_example=52, size_non_overlap=5):
    dataset_example_formatted = []
    example = []
    for value_armband in vector_to_format:
        if (example == []):
            example = value_armband
        else:
            example = np.row_stack((example, value_armband))
        if len(example) >= number_of_vector_per_example:
            example = example.transpose()
            dataset_example_formatted.append(example)
            example = example.transpose()
            example = example[size_non_overlap:]
    data_calculated = calculate_wavelet.calculate_wavelet_dataset(dataset_example_formatted)
    return np.array(data_calculated)


def read_data(path, modality="cwt"):
    print(os.walk(path))
    all_subjects_train_examples = []
    all_subjects_train_labels = []
    for subject_index in range(1, 11):
        paths_data = [path + 's' + str(subject_index) + '/S' + str(subject_index) + '_E2_A1.mat']
        train_examples = []
        train_labels = []
        for k, path_data in enumerate(paths_data):
            mat_contents = sio.loadmat(path_data)
            emg_data = mat_contents['emg'][:, 8::]  # Get the data from the second EMG
            labels_total = np.squeeze(mat_contents['restimulus'])
            print(np.shape(labels_total))

            dictionnary_regrouped_emg_repetitions = {}
            emg_big_window = []
            last_label = labels_total[0]
            for emg_entry, label_entry in zip(emg_data, labels_total):
                if label_entry != last_label:
                    if last_label in dictionnary_regrouped_emg_repetitions:
                        dictionnary_regrouped_emg_repetitions[last_label].append(emg_big_window)
                    else:
                        dictionnary_regrouped_emg_repetitions[last_label] = [emg_big_window]
                    emg_big_window = []

                last_label = label_entry
                emg_big_window.append(emg_entry)

            dictionnary_regrouped_emg_repetitions[last_label].append(emg_big_window)

            print(np.shape(dictionnary_regrouped_emg_repetitions[17]))

            keys_labels = dictionnary_regrouped_emg_repetitions.keys()
            print(keys_labels)
            if k == 0:
                number_class = len(labels_total)

            for key in keys_labels:
                for i, examples in enumerate(dictionnary_regrouped_emg_repetitions[key]):
                    if key == 0 and i > 5:
                        break
                    # Calculate the examples and labels according to the required window
                    if modality == "cwt":
                        dataset_example = format_data_to_train_cwt(examples)
                    else:
                        dataset_example = format_data_to_train_cwt(examples)
                    train_examples.append(dataset_example)
                    train_labels.append(key + np.zeros(len(dataset_example)))
        print(np.shape(train_examples))
        all_subjects_train_examples.append(train_examples)
        all_subjects_train_labels.append(train_labels)

    return all_subjects_train_examples, all_subjects_train_labels

# if __name__ == '__main__':
#     get_data('../../../data/ninaDB5/')

    #  Get data from E2 and E3
