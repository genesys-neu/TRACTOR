import argparse
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ORAN_dataset import *

import ray.train as train
from ray.train.torch import TorchTrainer, TorchPredictor
from ray.air.config import ScalingConfig
from ray.air import session, Checkpoint
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

#ds_train = ORANTracesDataset('train_in__Trial1_Trial2_Trial3.pkl', 'train_lbl__Trial1_Trial2_Trial3.pkl')
#ds_test = ORANTracesDataset('valid_in__Trial1_Trial2_Trial3.pkl', 'valid_lbl__Trial1_Trial2_Trial3.pkl')
# ds_train = ORANTracesDataset('train_in__Trial1_Trial2.pkl', 'train_lbl__Trial1_Trial2.pkl')
# ds_test = ORANTracesDataset('valid_in__Trial1_Trial2.pkl', 'valid_lbl__Trial1_Trial2.pkl')

ds_train = None
ds_test = None
Nclass = None
train_config = {"lr": 1e-3, "batch_size": 512, "epochs": 350}


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, classes=3):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(72, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define model
class ConvNN(nn.Module):
    def __init__(self, numChannels=1, slice_len=4, num_feats=18, classes=3):
        super(ConvNN, self).__init__()

        self.numChannels = numChannels

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(4, 1))
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))
        ##  initialize second set of CONV => RELU => POOL layers
        # self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
        #                    kernel_size=(5, 5))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## initialize first (and only) set of FC => RELU layers

        # pass a random input
        rand_x = torch.Tensor(np.random.random((1, slice_len, num_feats)))
        output_size = torch.flatten(self.conv1(rand_x)).shape
        self.fc1 = nn.Linear(in_features=output_size.numel(), out_features=512)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=512, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.reshape((x.shape[0], self.numChannels, x.shape[1], x.shape[2]))   # CNN 2D expects a [N, Cin, H, W] size of data
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)
        ## pass the output from the previous layer through the second
        ## set of CONV => RELU => POOL layers
        #x = self.conv2(x)
        #x = self.relu2(x)
        #x = self.maxpool2(x)
        ## flatten the output from the previous layer and pass it
        ## through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

global_model = ConvNN


def train_epoch(dataloader, model, loss_fn, optimizer, isDebug=False):
    if not isDebug:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

from sklearn.metrics import confusion_matrix as conf_mat
def validate_epoch(dataloader, model, loss_fn, Nclasses, isDebug=False):
    if not isDebug:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((Nclasses, Nclasses))
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            conf_matrix += conf_mat(y, pred.argmax(1), labels=list(range(Nclasses)))
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    pickle.dump(conf_matrix, open('conf_matrix.last.pkl', 'wb'))
    return test_loss


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    Nclass = config["Nclass"]
    isDebug = config['isDebug']
    slice_len = config['slice_len']
    num_feats = config['num_feats']

    if isDebug:
        worker_batch_size = batch_size
    else:
        worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders.
    train_dataloader = DataLoader(ds_train, batch_size=worker_batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=worker_batch_size, shuffle=True)

    if not isDebug:
        train_dataloader = train.torch.prepare_data_loader(train_dataloader)
        test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model.
    model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
    if not isDebug:
        model = train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []

    for e in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer, isDebug)
        loss = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, isDebug=isDebug)
        scheduler.step(loss)
        loss_results.append(loss)
        if not isDebug:

            # store checkpoint only if the loss has improved
            state_dict = model.state_dict()
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
            checkpoint = Checkpoint.from_dict(
                dict(epoch=e, model_weights=state_dict)
            )

            session.report(dict(loss=loss), checkpoint=checkpoint)

    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


def debug_train_func(config: Dict):
    num_epochs = config['epochs']
    Nclass = config["Nclass"]
    slice_len = config['slice_len']
    num_feats = config['num_feats']
    #model = NeuralNetwork()
    model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    worker_batch_size = config['batch_size']
    train_dataloader = DataLoader(ds_train, batch_size=worker_batch_size, shuffle=True)

    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # TODO create a batch
            output = model(inputs)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            np_labels = labels.cpu().detach().numpy()
            out_class_prob = torch.softmax(output, dim=1).cpu().detach().numpy()
            out_labels = np.argmax(out_class_prob, axis=1)
            accuracy = np.sum(np.equal(np_labels, out_labels))/np_labels.shape[0]
            losses.append(loss.item())
            accuracies.append(accuracy)
        loss_avg = sum(losses)/len(losses)
        accuracy_avg = sum(accuracies)/len(accuracies)
        print(f"epoch: {epoch}, loss: {loss.item()}, accuracy: {np.round(accuracy * 100, decimals=3)} %")


def train_ORAN_ds(num_workers=2, use_gpu=False):
    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    result = trainer.fit()
    print(f"Results: {result.metrics}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_file", required=True, help="Name of dataset pickle file containing training data and labels.")
    parser.add_argument("--ds_path", default="/home/mauro/Research/ORAN/traffic_gen2/logs/", help="Specify path where dataset files are stored")
    parser.add_argument("--isNorm", default=False, action='store_true', help="Specify to load the normalized dataset." )
    parser.add_argument("--isDebug", action='store_true', default=False, help="Run in debug mode")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model")
    args, _ = parser.parse_known_args()

    ds_train = ORANTracesDataset(args.ds_file, key='train', normalize=args.isNorm, path=args.ds_path)
    ds_test = ORANTracesDataset(args.ds_file, key='valid', normalize=args.isNorm, path=args.ds_path)

    ds_info = ds_train.info()

    Nclass = ds_info['nclasses']
    train_config['Nclass'] = Nclass
    train_config['isDebug'] = args.isDebug
    train_config['slice_len'] = ds_info['slice_len']
    train_config['num_feats'] = ds_info['numfeats']

    if not args.test:

        if not train_config['isDebug']:
            import ray
            ray.init(address=args.address)
            train_ORAN_ds(num_workers=args.num_workers, use_gpu=args.use_gpu)
        else:
            train_func(train_config)

        #debug_train_func(train_config)
    else:

        cp_path = '/home/mauro/ray_results/TorchTrainer_2022-12-07_17-09-41/TorchTrainer_d9111_00000_0_2022-12-07_17-09-41/checkpoint_000199/'
        model = global_model(classes=Nclass, slice_len=train_config['slice_len'], num_feats=train_config['num_feats'])
        cp = Checkpoint(local_path=cp_path)
        model.load_state_dict(cp.to_dict().get("model_weights"))
        dataloader = DataLoader(ds_test, batch_size=128)
        loss_fn = nn.CrossEntropyLoss()

        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        conf_matrix = np.zeros((Nclass, Nclass))
        mistaken_samples = []
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                #test_loss += loss_fn(pred, y).item()
                #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                labels = pred.argmax(1)
                # TODO find all the samples from MMTC and keep the ones that are misclassified as URLL
                MMTC_cix = 1
                mmtc_ixs = np.where(y == MMTC_cix)    # let's retrieve samples indexes in this batch that belong to the second class MMTC
                mmtc_labels = labels[mmtc_ixs]
                mmtc_in = X
                # find the indexes of all the mmtc samples that are misclassified as URLL
                URLL_cix = 2
                mmtc_label_eqURLL = np.where(mmtc_labels == URLL_cix)
                # get the sample indexes whose labels equals URLL_cix (although they are MMTC)
                wrong_mmtc_urll_ixs = np.array(mmtc_ixs)[0, mmtc_label_eqURLL]
                mistaken_samples.append(X[wrong_mmtc_urll_ixs, :, :])

        for m in mistaken_samples:
            for s in range(np.squeeze(m).shape[0]):
                plt.imshow(np.squeeze(m[:,s,:,:]))
                plt.colorbar()
                plt.show()




