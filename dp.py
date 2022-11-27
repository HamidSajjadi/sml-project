import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def df_to_dataset(my_x, my_y, batch_size):
    tensor_x = torch.Tensor(my_x.to_numpy())  # transform to torch tensor
    tensor_y = torch.Tensor(my_y.to_numpy())
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    return DataLoader(my_dataset, batch_size)  # cr


def get_model(n_input, n_hidden):
    assert len(n_hidden) >= 1
    modules = [nn.Linear(n_input, n_hidden[0]), nn.ReLU()]

    for idx, _ in enumerate(n_hidden[:-1]):
        modules.append(nn.Linear(n_hidden[idx], n_hidden[idx + 1]))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(n_hidden[-1], 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


def calculate_model_acc(m, test_data):
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        total, correct = 0, 0
        for _, data in enumerate(test, 0):
            # Get inputs
            inputs, targets = data

            # Generate outputs
            outputs = m(inputs)

            # Set total and correct
            predicted = (outputs > 0.5).int()
            total += targets.size(0)
            correct += (predicted.squeeze() == targets.squeeze()).sum().item()
    return correct / total


if __name__ == '__main__':
    # Configuration options

    k_folds = 5
    num_epochs = 100
    batch_size = 32
    loss_function = torch.nn.BCELoss()
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # For fold results
    results = {}

    df = pd.read_csv('data/train.csv')
    X = df.drop(columns="Lead")
    X = (X - X.mean()) / X.std()
    Y = df['Lead'].replace({"Male": 1, "Female": 0})

    model = get_model(13, [32, 16, 8, 4])

    # Set fixed random number seed
    torch.manual_seed(42)
    t = tqdm(enumerate(kfold.split(X)))
    for fold, (train_ids, test_ids) in t:
        x, y = X.iloc[train_ids], Y.iloc[train_ids]
        x_test, y_test = X.iloc[test_ids], Y.iloc[test_ids]
        train = df_to_dataset(x, y, batch_size)
        test = df_to_dataset(x_test, y_test, batch_size)

        reset_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Run the training loop for defined number of epochs
        epoch_t = tqdm(range(0, num_epochs))
        for epoch in epoch_t:
            # Print epoch

            # Set current loss value
            total_loss = 0.0
            train_correct, train_total = 0, 0
            for i, data in enumerate(train, 0):
                # Get inputs
                inputs, targets = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs.squeeze(), targets.squeeze())

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()
                predicted = (outputs > 0.5).int()
                train_total += targets.size(0)
                train_correct += (predicted.squeeze() == targets.squeeze()).sum().item()

                # Print statistics
                total_loss += loss.item()
                average_loss = total_loss / (i + 1)
                epoch_t.set_description(
                    f"train loss: {round(average_loss, 3)}, train_accuracy: {round(100.0 * (float(train_correct) / train_total), 3)}")

                # Saving the model
                save_path = f'./model-fold-{fold}.pth'
                torch.save(model.state_dict(), save_path)

            accuracy = calculate_model_acc(model, test)
            epoch_t.set_description(
                desc=f"train loss: {round(average_loss, 3)}, train_accuracy: {round(100.0 * (float(train_correct) / train_total), 3)}, test acc: {round(accuracy, 4)}")

        # Evaluation for this fold
        acc = calculate_model_acc(model, test)
        results[fold] = acc
        # Print fold results
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')
