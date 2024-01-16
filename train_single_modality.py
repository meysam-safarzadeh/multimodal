import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from model import SingleModalityTransformer
from torch.utils.data import DataLoader
from mint_pain_dataset_creator import create_dataset
from utils import class_wise_accuracy, plot_accuracy, plot_loss, load_checkpoint, FocalLoss
import torch.nn.functional as F


# Set random seed for reproducibility
random_seed = 41
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_folder, 'single_modality_model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, device, verbose, epoch, numEpochs, batch_size, train_size,
          modality):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_class_accuracies = []
    for i, (z1, z2, z3, labels) in enumerate(train_loader, 0):
        z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device),labels.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        if modality == 'fau':
            outputs = model(z1)
        elif modality == 'thermal':
            outputs = model(z2)
        elif modality == 'depth':
            outputs = model(z3)
        else:
            raise ValueError('Modality must be either "fau" or "thermal"')

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Calculate class-wise accuracy
        class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
        all_class_accuracies.append(class_accuracies)

    train_loss = running_loss / len(train_loader)

    # Average across all classes and batches for class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0)
    overall_avg_accuracy = np.mean(avg_class_accuracy) * 100

    # Print average loss and class-wise accuracy for the epoch
    if verbose:
        print('Epoch [%d/%d], Train Loss: %.4f, Average Class Accuracy: %.3f %%' %
              (epoch + 1, numEpochs, train_loss, overall_avg_accuracy))

    return train_loss, overall_avg_accuracy


def val(val_loader, model, criterion, device, verbose, epoch, numEpochs, batch_size, val_size, modality):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_class_accuracies = []

    with torch.no_grad():
        for i, (z1, z2, z3, labels) in enumerate(val_loader, 0):
            z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device), labels.to(device).long()

            if modality == 'fau':
                outputs = model(z1)
            elif modality == 'thermal':
                outputs = model(z2)
            elif modality == 'depth':
                outputs = model(z3)
            else:
                raise ValueError('Modality must be either "fau" or "thermal"')

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate class-wise accuracy
            class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
            all_class_accuracies.append(class_accuracies)

    val_loss = running_loss / len(val_loader)

    # Average across all classes and batches for class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0) * 100
    overall_avg_accuracy = np.mean(avg_class_accuracy)

    if verbose:
        print('Epoch [%d/%d], Validation Loss: %.3f' % (epoch + 1, numEpochs, val_loss))
        print('Validation Class-wise Accuracy:', np.round(avg_class_accuracy, 2))

    return val_loss, overall_avg_accuracy,  avg_class_accuracy


def test(test_loader, model, criterion, device, verbose, modality):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_outputs = []
    all_labels = []
    all_class_accuracies = []
    with torch.no_grad():
        for z1, z2, z3, labels in test_loader:
            z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device), labels.to(device)

            if modality == 'fau':
                outputs = model(z1)
            elif modality == 'thermal':
                outputs = model(z2)
            elif modality == 'depth':
                outputs = model(z3)
            else:
                raise ValueError('Modality must be either "fau" or "thermal"')

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Store outputs
            all_outputs.append(outputs.cpu())  # (batch_size, num_classes)
            all_labels.append(labels.cpu())  # (batch_size,)

            # Calculate class-wise accuracy
            class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
            all_class_accuracies.append(class_accuracies)

    test_loss = running_loss / len(test_loader)

    # Calculate class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0) * 100
    overall_avg_accuracy = np.mean(avg_class_accuracy)

    if verbose:
        print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, overall_avg_accuracy))
        print('Test Class-wise Accuracy:', np.round(avg_class_accuracy, 2))

    return test_loss, overall_avg_accuracy, torch.cat(all_outputs, axis=0), torch.cat(all_labels, axis=0)


def main(hidden_dim, num_heads, num_layers, learning_rate, dropout_rate, weight_decay, downsample_method, mode,
         fusion_layers, n_bottlenecks, batch_size, num_epochs, verbose, fold, device, save_model, max_seq_len,
         classification_head, plot, head_layer_sizes, output_dim, modality):
    """
        Main function for training an Attention-based Bottleneck Fusion model.

        Parameters:
        - hidden_dim: List of hidden dimensions for each modality and after fusion.
        - num_heads: Number of attention heads for each modality and after fusion.
        - num_layers: Number of transformer encoder layers for each modality.
        - learning_rate: Learning rate for the optimizer.
        - dropout_rate: Dropout rate used in the model.
        - weight_decay: Weight decay factor for the optimizer.
        - downsample_method: Method for downsampling (e.g., 'Linear', 'MaxPool').
        - mode: Mode of operation for the final classification layer ('concat' or 'separate').
        - fusion_layers: Number of layers after modality fusion.
        - n_bottlenecks: Number of bottleneck tokens in the model.
        - batch_size: Batch size for training and validation.
        - num_epochs: Number of epochs for training.
        - verbose: Verbosity mode.
        - fold: Fold number for cross-validation.
        - device: Device to use for training and validation.
        - save_model: Whether to save the model or not. If True, the model will be saved in the 'checkpoints' folder.
        - max_seq_len: Maximum sequence length for the input sequences. The length of the sequences + 1 CLS token
        - classification_head: Whether to use a classification head or not. If True, a classification head will be added.
        - plot: Whether to plot the loss and accuracy curves or not. bool = True or False
        - head_layer_sizes: List of hidden layer sizes for the classification head. 3 layers are used by default.
        - output_dim: Output of the transformer will be downsampled to this dimension before the classification head.
        - modality: Modality to use for training and validation. 'fau' or 'thermal' or 'depth'
    """
    # Initialize parameters and data
    input_dim_dic = {'fau': 22, 'thermal': 512, 'depth': 128}
    input_dim = input_dim_dic[modality]
    num_classes = 5

    # Initialize datasets and dataloaders
    # Paths to your files
    fau_file_path = 'embeddings_fau/FAU_embeddings_with_labels.csv'
    thermal_file_path = 'embeddings_thermal/Thermal_embeddings_and_filenames_new.npz'
    depth_file_path = 'embeddings_depth/Depth_embeddings_and_filenames_new.npz'
    split_file_path = 'cross_validation_split_2.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path,
                                                              fold, batch_size=batch_size, max_seq_len=max_seq_len,
                                                              depth_file_path=depth_file_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SingleModalityTransformer(input_dim, hidden_dim, num_heads, num_layers, fusion_layers, n_bottlenecks,
                                      num_classes, device, max_seq_len+1, mode, dropout_rate,
                                      downsample_method, classification_head, head_layer_sizes, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = float(0.0)
    for epoch in range(num_epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, False,
                                      epoch, num_epochs, batch_size, len(train_dataset), modality)
        val_loss, val_acc, class_wise_acc = val(val_loader, model, criterion, device, False,
                                               epoch, num_epochs, batch_size, len(val_dataset), modality)
        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',
                  f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')
            print('Validation Class-wise Accuracy:', np.round(class_wise_acc, 2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save checkpoints
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            if save_model:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
                torch.save(model, 'checkpoints/model_best_' + modality + 'Only.pth')
                if verbose:
                    print("Checkpoint saved: Epoch {}, Validation Accuracy {}".format(epoch + 1, best_val_acc))

    # Load the best model and test the model based on that
    # model, _, _, _ = load_checkpoint(model, optimizer, 'checkpoints/model_best.pth.tar')
    # test(test_loader, model, criterion, device, True, modality)

    if plot:
        plot_loss(train_losses, val_losses, 'loss_curve.png')
        plot_accuracy(train_accuracies, val_accuracies, 'accuracy_curve.png')

    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc


def ensemble_single_modalities(hidden_dim, num_heads, num_layers, learning_rate, dropout_rate, weight_decay, downsample_method, mode,
                               fusion_layers, n_bottlenecks, batch_size, num_epochs, verbose, fold, device, save_model, max_seq_len,
                               classification_head, plot, head_layer_sizes, output_dim, modality):
    """
        Main function for training an Attention-based Bottleneck Fusion model.

        Parameters:
        - hidden_dim: List of hidden dimensions for each modality and after fusion.
        - num_heads: Number of attention heads for each modality and after fusion.
        - num_layers: Number of transformer encoder layers for each modality.
        - learning_rate: Learning rate for the optimizer.
        - dropout_rate: Dropout rate used in the model.
        - weight_decay: Weight decay factor for the optimizer.
        - downsample_method: Method for downsampling (e.g., 'Linear', 'MaxPool').
        - mode: Mode of operation for the final classification layer ('concat' or 'separate').
        - fusion_layers: Number of layers after modality fusion.
        - n_bottlenecks: Number of bottleneck tokens in the model.
        - batch_size: Batch size for training and validation.
        - num_epochs: Number of epochs for training.
        - verbose: Verbosity mode.
        - fold: Fold number for cross-validation.
        - device: Device to use for training and validation.
        - save_model: Whether to save the model or not. If True, the model will be saved in the 'checkpoints' folder.
        - max_seq_len: Maximum sequence length for the input sequences. The length of the sequences + 1 CLS token
        - classification_head: Whether to use a classification head or not. If True, a classification head will be added.
        - plot: Whether to plot the loss and accuracy curves or not. bool = True or False
        - head_layer_sizes: List of hidden layer sizes for the classification head. 3 layers are used by default.
        - output_dim: Output of the transformer will be downsampled to this dimension before the classification head.
        - modality: Modality to use for training and validation. 'fau' or 'thermal' or 'depth'
    """
    # Initialize parameters and data
    input_dim_dic = {'fau': 22, 'thermal': 512, 'depth': 128}
    input_dim = input_dim_dic[modality]
    num_classes = 5

    # Initialize datasets and dataloaders
    # Paths to your files
    fau_file_path = 'embeddings_fau/FAU_embeddings_with_labels.csv'
    thermal_file_path = 'embeddings_thermal/Thermal_embeddings_and_filenames_new.npz'
    depth_file_path = 'embeddings_depth/Depth_embeddings_and_filenames_new.npz'
    split_file_path = 'cross_validation_split_2.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path,
                                                              fold, batch_size=batch_size, max_seq_len=max_seq_len,
                                                              depth_file_path=depth_file_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    fau_model = torch.load('checkpoints/model_best_fauOnly.pth')
    thermal_model = torch.load('checkpoints/model_best_thermalOnly.pth')
    depth_model = torch.load('checkpoints/model_best_depthOnly.pth')

    criterion = nn.CrossEntropyLoss()
    _, _, output_fau, labels = test(val_loader, fau_model, criterion, device, True, 'fau')
    _, _, output_depth, _ = test(val_loader, depth_model, criterion, device, True, 'depth')
    _, _, output_thermal, _ = test(val_loader, thermal_model, criterion, device, True, 'thermal')
    print(output_depth.shape)
    # output_thermal = F.softmax(output_thermal, dim=1)
    # output_depth = F.softmax(output_depth, dim=1)
    # output_fau = F.softmax(output_fau, dim=1)
    # output_fau[:, 0] *= 100
    # output_fau[:, 4] *= 100
    # output_fau[:, 2] *= 100
    _, predicted_fau = torch.max(output_fau, 1)
    _, predicted_depth = torch.max(output_depth, 1)
    _, predicted_thermal = torch.max(output_thermal, 1)

    # output_depth[:, 3] *= 100
    # output_thermal[:, 1] *= 100


    predicted = (predicted_depth + predicted_fau)/2
    _, predicted = torch.max(final_output, 1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, np.round(predicted), labels=np.arange(num_classes))
    cm2 = confusion_matrix(labels[predicted_fau != 0], predicted_depth[predicted_fau != 0], labels=np.arange(num_classes))
    cm3 = confusion_matrix(labels[predicted_fau != 0], predicted_thermal[predicted_fau != 0], labels=np.arange(num_classes))

    # cm3 = confusion_matrix(labels[predicted_fau != 0 & predicted_depth != 0],
                           # predicted_thermal[predicted_fau != 0 & predicted_depth != 0], labels=np.arange(num_classes))

    predict_final = np.round((predicted_depth + predicted_thermal + predicted_fau)/3)
    class_accuracies = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # class_accuracies = class_wise_accuracy(final_output, labels, 5)
    print(class_accuracies, np.mean(class_accuracies))

    # 0.568 * (187+36+39+35+32)
    # 329 +


if __name__ == '__main__':
    # for i in range(5):
    # _, _, _, _, best_val_acc = main(hidden_dim=92, num_heads=2, num_layers=2, learning_rate=0.00011,
    #                                 dropout_rate=0.0, weight_decay=0.0, downsample_method='Linear', mode=None,
    #                                 fusion_layers=None, n_bottlenecks=None, batch_size=256, num_epochs=200, verbose=True, fold=1,
    #                                 device='cuda:1', save_model=True, max_seq_len=36, classification_head=True, plot=True,
    #                                 head_layer_sizes=[64, 128, 64], output_dim=22, modality='fau')
    # print('Fold: {}, Best Validation Accuracy: {}'.format(i, best_val_acc))


    ensemble_single_modalities(hidden_dim=96, num_heads=2, num_layers=5, learning_rate=3e-4,
                               dropout_rate=0.0, weight_decay=0.0, downsample_method=None, mode=None,
                               fusion_layers=None, n_bottlenecks=None, batch_size=64, num_epochs=150, verbose=True, fold=1,
                               device='cuda:1', save_model=True, max_seq_len=36, classification_head=True, plot=True,
                               head_layer_sizes=[352, 112, 48], output_dim=22, modality='depth')
    # output_dim=22 for FAU
    # output_dim=32 for thermal

    # _, _, _, _, _ = main(hidden_dim=96, num_heads=2, num_layers=5, learning_rate=3e-4,

