import argparse
from typing import Dict
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ORAN_dataset import *

import ray.train as train
from ray.train.torch import TorchTrainer, TorchPredictor
from ray.air.config import ScalingConfig
from ray.air import session, Checkpoint
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from ORAN_dataset import load_csv_traces

from sklearn.metrics import confusion_matrix as conf_mat
import seaborn as sn

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

class TransformerNN(nn.Module):
    def __init__(self, classes: int = 4, num_feats: int = 18, slice_len: int = 32, nhead: int = 1, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False):
        super(TransformerNN, self).__init__()
        self.norm = nn.LayerNorm(num_feats)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(num_feats + 1, dropout)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(num_feats, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats*slice_len, num_feats)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(num_feats, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        #src = self.norm(src) should not be necessary since output can be already normalized
        # apply positional encoding if decided
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # ToDo: try the following change
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        # try the following instead
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :-1]
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, features]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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
    global_model = config['global_model']

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

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []
    
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL                {total_params}')

    best_loss = np.inf
    epochs_wo_improvement = 0
    for e in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer, isDebug)
        loss = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, isDebug=isDebug)
        scheduler.step(loss)
        loss_results.append(loss)
        epochs_wo_improvement += 1
        if not isDebug:

            # store checkpoint only if the loss has improved
            state_dict = model.state_dict()
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
            checkpoint = Checkpoint.from_dict(
                dict(epoch=e, model_weights=state_dict)
            )

            session.report(dict(loss=loss), checkpoint=checkpoint)
        else:
            if best_loss > loss:
                epochs_wo_improvement = 0
                best_loss = loss
                model_name = f'model.{slice_len}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join('./', model_name))

        if epochs_wo_improvement > 10: # early stopping
            return loss_results
    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


def debug_train_func(config: Dict):
    num_epochs = config['epochs']
    Nclass = config["Nclass"]
    slice_len = config['slice_len']
    num_feats = config['num_feats']
    global_model = config['global_model']
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


def timing_inference_GPU(dummy_input, model):
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_file", required=True, help="Name of dataset pickle file containing training data and labels.")
    parser.add_argument("--ds_path", default="/home/mauro/Research/ORAN/traffic_gen2/logs/", help="Specify path where dataset files are stored")
    parser.add_argument("--isNorm", default=False, action='store_true', help="Specify to load the normalized dataset." )
    parser.add_argument("--isDebug", action='store_true', default=False, help="Run in debug mode")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model") # TODO visualize capture and then perform classification after loading model
    parser.add_argument("--cp_path", help='Path to the checkpoint to load at test time.')
    parser.add_argument("--norm_param_path", default="/home/mauro/Research/ORAN/traffic_gen2/logs/cols_maxmin.pkl", help="normalization parameters path.")
    parser.add_argument("--useTransformer", action="store_true", default=False, help="Use Transformer based model instead of CNN")
    args, _ = parser.parse_known_args()

    ds_train = ORANTracesDataset(args.ds_file, key='train', normalize=args.isNorm, path=args.ds_path)
    ds_test = ORANTracesDataset(args.ds_file, key='valid', normalize=args.isNorm, path=args.ds_path)

    ds_info = ds_train.info()

    Nclass = ds_info['nclasses']
    train_config['Nclass'] = Nclass
    train_config['isDebug'] = args.isDebug
    train_config['slice_len'] = ds_info['slice_len']
    train_config['num_feats'] = ds_info['numfeats']
    train_config['global_model'] = TransformerNN if args.useTransformer else ConvNN

    if not args.test:

        if not train_config['isDebug']:
            import ray
            ray.init(address=args.address)
            train_ORAN_ds(num_workers=args.num_workers, use_gpu=args.use_gpu)
        else:
            train_func(train_config)

        #debug_train_func(train_config)
    else:
        cp_path = args.cp_path
        global_model = TransformerNN if args.useTransformer else ConvNN
        model = global_model(classes=Nclass, slice_len=train_config['slice_len'], num_feats=train_config['num_feats'])
        cp = Checkpoint(local_path=cp_path)
        model.load_state_dict(cp.to_dict().get("model_weights"))

        # save a new model state using torch functions to allow loading model into gpu
        device = torch.device("cuda")
        os.makedirs('model',exist_ok=True)
        model_params_filename =  os.path.join('model', 'model_weights__slice'+str(train_config['slice_len'])+'.pt')
        torch.save(model.state_dict(), model_params_filename)
        model.load_state_dict(torch.load(model_params_filename, map_location='cuda:0'))
        model.to(device)

        """
        dataloader = DataLoader(ds_test, batch_size=128)

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
                
                #  find all the samples from MMTC and keep the ones that are misclassified as URLL
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
        """

        # load normalization parameters
        colsparam_dict = pickle.load(open(args.norm_param_path, 'rb'))
        # load and normalize a trace input
        traces = load_csv_traces(['Trial1', 'Trial2', 'Trial3', 'Trial4', 'Trial5', 'Trial6'], args.ds_path, norm_params=colsparam_dict)
        classmap = {'embb': 0, 'mmtc': 1, 'urll': 2, 'ctrl': 3}
        y_true = []
        y_output = []

        for tix, dtrace in enumerate(traces):
            print('------------ Trial', tix+1, '------------')
            trial_y_true = []
            trial_y_output = []
            for k in dtrace.keys():
                num_correct = 0
                tr = dtrace[k].values
                correct_class = classmap[k]

                for t in range(tr.shape[0]):
                    if t + train_config['slice_len'] < tr.shape[0]:
                        input_sample = tr[t:t + train_config['slice_len']]
                        input = torch.Tensor(input_sample[np.newaxis, :, :])
                        input = input.to(device)    # transfer input data to GPU
                        pred = model(input)
                        class_ix = pred.argmax(1)
                        co = class_ix.cpu().numpy()[0]
                        #if co == correct_class:
                        #    num_correct += 1
                        y_true.append(correct_class)
                        y_output.append(co)
                        trial_y_true.append(correct_class)
                        trial_y_output.append(co)

                    #mean, stddev = timing_inference_GPU(input, model)
                #print('[',k,'] Correct % ', num_correct/tr.shape[0]*100)
            trial_cm = conf_mat(trial_y_true, trial_y_output, labels=list(range(len(dtrace.keys()))))
            for r in range(trial_cm.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(trial_cm[r, :])
                trial_cm[r, :] = trial_cm[r, :] / sum_row * 100.  # compute in percentage
            print('Confusion Matrix (%)')
            print(trial_cm)

        cm = conf_mat(y_true,y_output, labels=list(range(len(classmap.keys()))))

        for r in range(cm.shape[0]):  # for each row in the confusion matrix
            sum_row = np.sum(cm[r, :])
            cm[r, :] = cm[r, :] / sum_row  * 100.# compute in percentage


        axis_lbl = ['eMBB', 'mMTC', 'URLLC'] if cm.shape[0] == 3 else ['eMBB', 'mMTC', 'URLLC', 'ctrl']
        df_cm = pd.DataFrame(cm, axis_lbl, axis_lbl)
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.show()
        plt.clf()
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Global confusion matrix (%) (all trials)')
        print(cm)

        #plt.figure(figsize=(30, 1))
        #plt.imshow(tr[:1000, :].T)
        #plt.show()













