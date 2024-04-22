import argparse
import os
import pickle

from preprocessing import SyscallFileReader


def build_vocabulary(syscalls, syscall_vocab):
    # Build vocabulary from the list of syscalls
    for syscall in syscalls:
        if syscall not in syscall_vocab:
            # Assign a new index to the syscall
            syscall_vocab[syscall] = len(syscall_vocab)
    return syscall_vocab


def process_subfolder(reader, subfolder_path, custom_label, syscall_vocab, tokenize=False):
    subfolder_sequence_data = []
    max_seq_len = 0

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)
        print(file_path)
        syscalls = reader.read(file_path)

        # Build or update vocabulary with syscalls from this file
        syscall_vocab = build_vocabulary(syscalls, syscall_vocab)

        # Convert syscalls to their corresponding token values from the vocabulary
        if tokenize:
            syscalls = [syscall_vocab[syscall] for syscall in syscalls]
        max_seq_len = max(max_seq_len, len(syscalls))  # Update the maximum length

        # Append tokenized sequence and label to the list
        label = custom_label if custom_label else os.path.basename(subfolder_path)
        subfolder_sequence_data.append({'sequence': syscalls, 'label': label})

    return subfolder_sequence_data, syscall_vocab, max_seq_len


def preprocess_dataset(dataset_folder, filter_calls, output_filename):
    all_sequence_data = []
    reader = SyscallFileReader(filter_calls)
    syscall_vocab = {}
    global_max_seq_len = 0

    main_dirs = ['Attack_Data_Master', 'Training_Data_Master', 'Validation_Data_Master']
    for subfolder_name in os.listdir(dataset_folder):
        subfolder_path = os.path.join(dataset_folder, subfolder_name)
        if os.path.isdir(subfolder_path) and subfolder_name in main_dirs:
            if subfolder_name == 'Attack_Data_Master':
                for attack_type in os.listdir(subfolder_path):
                    attack_subfolder_path = os.path.join(subfolder_path, attack_type)
                    label = f'attack_{attack_type}'
                    data, syscall_vocab, max_len = process_subfolder(reader, attack_subfolder_path, label, syscall_vocab)
                    all_sequence_data.extend(data)
                    global_max_seq_len = max(global_max_seq_len, max_len)
            else:
                label = 'normal'
                data, syscall_vocab, max_len = process_subfolder(reader, subfolder_path, label, syscall_vocab)
                all_sequence_data.extend(data)
                global_max_seq_len = max(global_max_seq_len, max_len)

    # Serialize all sequence data and vocabulary to a single file
    output_file_path = os.path.join(dataset_folder, output_filename)
    with open(output_file_path, 'wb') as f:
        data_to_save = {
            'sequence_data': all_sequence_data,
            'vocab_size': len(syscall_vocab),
            'vocab': syscall_vocab,
            'max_seq_len': global_max_seq_len
        }
        pickle.dump(data_to_save, f)

    return output_file_path


def load_and_print_dataset(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    sequence_data = loaded_data['sequence_data']
    vocab_size = loaded_data['vocab_size']
    syscall_vocab = loaded_data['vocab']
    max_seq_len = loaded_data['max_seq_len']
    for item in sequence_data:
        print(f"Sequence: {item['sequence']}\nLabel: {item['label']}\n")
    print(f"Vocabulary Size: {vocab_size}")
    print('Vocab: ', syscall_vocab)
    print(f"Max Length: {max_seq_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset of system call logs and generate sequence representations.')
    parser.add_argument('dataset_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('--filter', action='store_true', help='Filter to include only relevant system calls')
    parser.add_argument('--output', type=str, default='processed_sequences.pkl', help='Output file for processed sequences')

    args = parser.parse_args()

    output_file_path = preprocess_dataset(args.dataset_folder, args.filter, args.output)
    load_and_print_dataset(output_file_path)
