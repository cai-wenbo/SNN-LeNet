import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SNN_LeNet
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import json
import os
import argparse


def load_model(model_path_src):
    model = SNN_LeNet()
    if os.path.exists(training_config['model_path_src']):
        model_dict = torch.load(training_config['model_path_src'])
        model.load_state_dict(model_dict)

    return model

def load_trails(training_config):
    step_losses = list()
    if os.path.exists(training_config['step_losses_pth']):
        with open(training_config['step_losses_pth'], 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    if os.path.exists(training_config['train_losses_pth']):
        with open(training_config['train_losses_pth'], 'r') as file:
            train_losses = json.load(file)
            file.close()
    
    test_losses = list()
    if os.path.exists(training_config['test_losses_pth']):
        with open(training_config['test_losses_pth'], 'r') as file:
            test_losses = json.load(file)
            file.close()

    train_accuracy = list()
    if os.path.exists(training_config['train_accuracy_pth']):
        with open(training_config['train_accuracy_pth'], 'r') as file:
            train_accuracy = json.load(file)
            file.close()
    
    test_accuracy = list()
    if os.path.exists(training_config['test_accuracy_pth']):
        with open(training_config['test_accuracy_pth'], 'r') as file:
            test_accuracy = json.load(file)
            file.close()

    return step_losses, train_losses, test_losses, train_accuracy, test_accuracy


def train_test_loop(training_config, model, dataloader_train, dataloader_test, optimizer, creterian, step_losses, train_losses, test_losses, train_accuracy, test_accuracy, device):
    encoder = encoding.PoissonEncoder()
    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        correct        = 0
        model.train()
        #  train loop
        for i, batch in enumerate(dataloader_train):
            batch = tuple(t.to(device) for t in batch)
            b_inputs, b_labels = batch
            out_fr = torch.zeros((b_inputs.shape[0], 10))


            optimizer.zero_grad()
            for j in range(training_config['time_out']):
                b_inputs_encoded = encoder(b_inputs)
                out_fr += model(b_inputs)
            
            #  reset the model
            functional.reset_net(model)

            out_fr /= training_config['time_out']
            loss = creterian(out_fr, b_labels)
            loss.backward()
            optimizer.step()
            loss_scalar = loss.item()
            loss_sum_train += loss_scalar
            step_losses.append(loss_scalar)

            b_predicts = torch.argmax(out_fr, dim=-1)
            correct += (b_predicts == b_labels).sum().item()

        train_loss = loss_sum_train / len(dataloader_train)
        train_losses.append(train_loss)
        train_acc = correct / len(dataloader_train.dataset)
        train_accuracy.append(train_acc)


        loss_sum_test = 0
        correct = 0

        model.eval() 
        #  test_loop
        for i, batch in enumerate(dataloader_test):
            batch = tuple(t.to(device) for t in batch)
            b_inputs, b_labels = batch
            out_fr = torch.zeros((b_inputs.shape[0], 10))

            with torch.no_grad():
                for j in range(training_config['time_out']):
                    b_inputs_encoded = encoder(b_inputs)
                    out_fr += model(b_inputs)
                
                #  reset the model
                functional.reset_net(model)

                out_fr /= training_config['time_out']
                loss = creterian(out_fr, b_labels)
                loss_scalar = loss.item()
                loss_sum_test += loss_scalar
                step_losses.append(loss_scalar)

                b_predicts = torch.argmax(out_fr, dim=-1)
                correct += (b_predicts == b_labels).sum().item()

        test_loss = loss_sum_test / len(dataloader_test)
        test_losses.append(test_loss)
        test_acc = correct / len(dataloader_test.dataset)
        test_accuracy.append(test_acc)



        print(f'Epoch: {epoch+1} \n Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.6f} \n train Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.6f}')



def save_trails(training_config, step_losses, train_losses, test_losses, train_accuracy, test_accuracy):
    with open(training_config['step_losses_pth'], 'w') as file:
        json.dump(step_losses, file)
        file.close()

    with open(training_config['train_losses_pth'], 'w') as file:
        json.dump(train_losses, file)
        file.close()
    
    with open(training_config['test_losses_pth'], 'w') as file:
        json.dump(test_losses, file)
        file.close()

    with open(training_config['train_accuracy_pth'], 'w') as file:
        json.dump(train_accuracy, file)
        file.close()
    
    with open(training_config['test_accuracy_pth'], 'w') as file:
        json.dump(test_accuracy, file)
        file.close()


def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    model
    load model and the history
    '''
    model = load_model(training_config['model_path_src']).to(device)

    #  load the losses history
    step_losses, train_losses, test_losses, train_accuracy, test_accuracy = load_trails(training_config)



    '''
    dataloader
    '''
    train_dataset = mnist.MNIST(root = '.', train = True, transform  = ToTensor(), download = True)
    test_dataset  = mnist.MNIST(root = '.', train = False, transform = ToTensor(), download = True)

    dataloader_train = DataLoader(train_dataset, batch_size = training_config['batch_size'], shuffle = True)
    dataloader_test  = DataLoader(test_dataset, batch_size  = training_config['batch_size'], shuffle = False)


    '''
    optimizer
    '''
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'], weight_decay = training_config['weight_decay'])



    '''
    creterian
    '''
    creterian = nn.CrossEntropyLoss()


    '''
    train  and validate loops
    '''
    train_test_loop(training_config, model, dataloader_train, dataloader_test, optimizer, creterian, step_losses, train_losses, test_losses, train_accuracy, test_accuracy, device)


        
    '''    
    save model and data
    '''

    model = model.to('cpu')
    torch.save(model.state_dict(), training_config['model_path_dst'])

    #  save the loss of the steps
    save_trails(training_config, step_losses, train_losses, test_losses, train_accuracy, test_accuracy)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs"    , type=int   , help="number of epochs"                                  , default=20)
    parser.add_argument("--batch_size"       , type=int   , help="batch size"                                        , default=512)
    parser.add_argument("--learning_rate"    , type=float  , help="learning rate"                                    , default=5e-4)
    parser.add_argument("--weight_decay"     , type=float  , help="weight_decay"                                     , default=1e-4)
    parser.add_argument("--vocab_size"       , type=int    , help="vocab size"                                       , default=21128)
    parser.add_argument("--embedding_dim"    , type=int    , help="embedding dimmention"                             , default=512)
    parser.add_argument("--LSTM_hidden_size" , type=int    , help="hidden_size of the BiLSTM model"                  , default=256)
    parser.add_argument("--LSTM_num_layers"  , type  = int , help   = "num_layers of the BiLSTM model"               , default               = 1)
    parser.add_argument("--num_labels"       , type  = int , help   = "types of labels"                              , default               = 6)
    parser.add_argument("--time_out"         , type  = int , help   = "maximum length of the time sequence"          , default               = 128)
    parser.add_argument("--model_path_dst"   , type  = str , help   = "the directory to save model"                  , default               = './saved_models/saved_dict.pth')
    parser.add_argument("--model_path_src"   , type = str  , help = "the directory to load model"                    , default = './saved_models/saved_dict.pth')
    parser.add_argument("--step_losses_pth"  , type=str    , help="the path of the json file that saves step losses" , default='./trails/step_losses.json')
    parser.add_argument("--train_losses_pth" , type=str   , help="the path of the json file that saves train losses" , default='./trails/train_losses.json')
    parser.add_argument("--test_losses_pth"  , type=str   , help="the path of the json file that saves test losses"  , default='./trails/test_losses.json')
    parser.add_argument("--train_accuracy_pth" , type=str   , help="the path of the json file that saves train accuracy" , default='./trails/train_accuracy.json')
    parser.add_argument("--test_accuracy_pth"  , type=str   , help="the path of the json file that saves test accuracy"  , default='./trails/test_accuracy.json')

    
    args = parser.parse_args()

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)
