import argparse
from typing import Dict
import matplotlib.pyplot as plt
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from ORAN_dataset import *

import ray.train as train
from ray.train.torch import TorchTrainer, TorchPredictor
from ray.air.config import ScalingConfig
from ray.air import session, Checkpoint
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from ORAN_dataset import load_csv_traces

from sklearn.metrics import confusion_matrix as conf_mat
import seaborn as sn

import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
import sys
sys.path.append(proj_root_dir)
from ORAN_models import ConvNN, TransformerNN, TransformerNN_v2
from visualize_inout import plot_trace_class

#ds_train = ORANTracesDataset('train_in__Trial1_Trial2_Trial3.pkl', 'train_lbl__Trial1_Trial2_Trial3.pkl')
#ds_test = ORANTracesDataset('valid_in__Trial1_Trial2_Trial3.pkl', 'valid_lbl__Trial1_Trial2_Trial3.pkl')
# ds_train = ORANTracesDataset('train_in__Trial1_Trial2.pkl', 'train_lbl__Trial1_Trial2.pkl')
# ds_test = ORANTracesDataset('valid_in__Trial1_Trial2.pkl', 'valid_lbl__Trial1_Trial2.pkl')

ds_train = None
ds_test = None
Nclass = None
train_config = {"lr": 1e-3, "batch_size": 512, "epochs": 350}

def train_epoch(dataloader, model, loss_fn, optimizer, isDebug=False):
    if not isDebug:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    model.train()
    start_time = time.time()
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
    print("--- %s seconds ---" % (time.time() - start_time))
    return (time.time() - start_time)

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


def train_func(config: Dict, check_zeros: bool):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    Nclass = config["Nclass"]
    isDebug = config['isDebug']
    slice_len = config['slice_len']
    num_feats = config['num_feats']
    global_model = config['global_model']
    model_postfix = config['model_postfix']

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
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=6, min_lr=0.00001, verbose=True)
    loss_results = []
    
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL                {total_params}')

    best_loss = np.inf
    epochs_wo_improvement = 0
    times = []
    for e in range(epochs):
        ep_time = train_epoch(train_dataloader, model, loss_fn, optimizer, isDebug)
        times.append(ep_time)
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
                ctrl_suffix = '.ctrl' if check_zeros else ''
                model_name = f'model.{slice_len}.{model_postfix}{ctrl_suffix}.pt'
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss,
                # }, os.path.join('./', model_name))

        if epochs_wo_improvement > 12: # early stopping
            print('------------------------------------')
            print('Early termination implemented at epoch:', e+1)
            print('------------------------------------')
            print(f'Training time analysis for model {config["model_postfix"]} with slice {slice_len}:')
            sd = np.std(times)
            mean = np.mean(times)
            print(f'Mean: {mean}, std: {sd}')
            with open('tr_time.txt', 'a') as f:
                f.write(f'Model {config["model_postfix"]} with slice {slice_len}:\n')
                f.write(f'Mean: {mean}, std: {sd}, num. epochs: {e+1}\n')
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
    parser.add_argument("--test", default=None, choices=['val', 'traces'], help="Testing the model") # TODO visualize capture and then perform classification after loading model
    parser.add_argument("--relabel_test", action="store_true", default=False, help="Perform ctrl label correction during testing time") 
    parser.add_argument("--cp_path", help='Path to the checkpoint to load at test time.')
    parser.add_argument("--norm_param_path", default="/home/mauro/Research/ORAN/traffic_gen2/logs/cols_maxmin.pkl", help="normalization parameters path.")
    parser.add_argument("--transformer", default=None, choices=['v1', 'v2'], help="Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)")
    args, _ = parser.parse_known_args()
    if args.transformer is not None:
        transformer = TransformerNN if args.transformer == 'v1' else TransformerNN_v2
    ds_train = ORANTracesDataset(args.ds_file, key='train', normalize=args.isNorm, path=args.ds_path)
    ds_test = ORANTracesDataset(args.ds_file, key='valid', normalize=args.isNorm, path=args.ds_path)

    ds_info = ds_train.info()

    Nclass = ds_info['nclasses']
    train_config['Nclass'] = Nclass
    train_config['isDebug'] = args.isDebug
    train_config['slice_len'] = ds_info['slice_len']
    train_config['num_feats'] = ds_info['numfeats']
    if args.transformer is None:
        train_config['global_model'] = ConvNN
    else:
        train_config['global_model'] = transformer
    train_config['model_postfix'] = 'trans_' + args.transformer if args.transformer is not None else 'cnn'

    if args.test is None:
        check_zeros = args.ds_file.split('_')[-2] == 'ctrlcorrected'
        if not train_config['isDebug']:
            import ray
            ray.init(address=args.address)
            train_ORAN_ds(num_workers=args.num_workers, use_gpu=args.use_gpu)
        else:
            train_func(train_config, check_zeros)

        #debug_train_func(train_config)
    else: 
        cp_path = args.cp_path
        check_zeros = args.relabel_test
        ctrl_flag = cp_path.split('/')[-1].split('.')[-1] == 'ctrl' # check if model trained under label correction

        global_model = train_config['global_model']
        model = global_model(classes=Nclass, slice_len=train_config['slice_len'], num_feats=train_config['num_feats'])
        device = torch.device("cuda")
        if args.isDebug:
            model.load_state_dict(torch.load(cp_path, map_location='cuda:0')['model_state_dict'])
        else:
            cp = Checkpoint(local_path=cp_path)
            model.load_state_dict(cp.to_dict().get("model_weights"))
            # save a new model state using torch functions to allow loading model into gpu
            os.makedirs('model',exist_ok=True)
            model_params_filename =  os.path.join('model', 'model_weights__slice'+str(train_config['slice_len'])+'.pt')
            torch.save(model.state_dict(), model_params_filename)
            model.load_state_dict(torch.load(model_params_filename, map_location='cuda:0'))
        model.to(device)
        model.eval()

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
        if args.test == 'traces':
            # load normalization parameters
            colsparam_dict = pickle.load(open(args.norm_param_path, 'rb'))
            # load and normalize a trace input
            traces = load_csv_traces(['Trial1', 'Trial2', 'Trial3', 'Trial4', 'Trial5', 'Trial6'], args.ds_path, norm_params=colsparam_dict)
            classmap = {'embb': 0, 'mmtc': 1, 'urll': 2, 'ctrl': 3}
            y_true = []
            y_output = []
            with torch.no_grad():
                for tix, dtrace in enumerate(traces):
                    print('------------ Trial', tix+1, '------------')
                    trial_y_true = []
                    trial_y_output = []
                    for k in dtrace.keys():
                        num_correct = 0
                        tr = dtrace[k].values
                        #correct_class = classmap[k]
                        output_list_kpi = []
                        output_list_y = []
                        for t in range(tr.shape[0]):
                            if t + train_config['slice_len'] < tr.shape[0]:
                                input_sample = tr[t:t + train_config['slice_len']]
                                input = torch.Tensor(input_sample[np.newaxis, :, :])
                                input = input.to(device)    # transfer input data to GPU
                                pred = model(input)
                                class_ix = pred.argmax(1)
                                correct_class = classmap[k]
                                if check_zeros:
                                    zeros = (input_sample == 0).astype(int).sum(axis=1)
                                    if (zeros > 10).all():
                                        correct_class = 3 # control if all KPIs rows have > 10 zeros
                                co = class_ix.cpu().numpy()[0]
                                #if co == correct_class:
                                #    num_correct += 1
                                y_true.append(correct_class)
                                y_output.append(co)
                                trial_y_true.append(correct_class)
                                trial_y_output.append(co)
                                output_list_kpi.append(tr[t])
                                output_list_y.append(co)

                            #mean, stddev = timing_inference_GPU(input, model)
                        #print('[',k,'] Correct % ', num_correct/tr.shape[0]*100)
                        assert (len(output_list_kpi) == len(output_list_y))
                        plot_trace_class(output_list_kpi, output_list_y,
                                        'traces_pdf_train_slice' + str(train_config['slice_len']) + '/trial' + str(
                                            tix + 1) + '_' + k, train_config['slice_len'], head=len(output_list_kpi), save_plain=True)
                    trial_cm = conf_mat(trial_y_true, trial_y_output, labels=list(range(len(classmap.keys()))))

                    for r in range(trial_cm.shape[0]):  # for each row in the confusion matrix
                        sum_row = np.sum(trial_cm[r, :])
                        if sum_row == 0:
                            sum_row = 1
                        trial_cm[r, :] = trial_cm[r, :] / sum_row * 100.  # compute in percentage
                    print('Confusion Matrix (%)')
                    print(trial_cm)

            cm = conf_mat(y_true,y_output, labels=list(range(len(classmap.keys()))))
            cm = cm.astype('float')
            for r in range(cm.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(cm[r, :])
                cm[r, :] = cm[r, :] / sum_row  * 100.# compute in percentage


            axis_lbl = ['eMBB', 'mMTC', 'URLLC'] if cm.shape[0] == 3 else ['eMBB', 'mMTC', 'URLLC', 'ctrl']
            df_cm = pd.DataFrame(cm, axis_lbl, axis_lbl)
            # plt.figure(figsize=(10,7))
            sn.set(font_scale=1.4)  # for label size
            sn.heatmap(df_cm, vmin=0, vmax=100, annot=True, cmap=sn.color_palette("light:b_r", as_cmap=True), annot_kws={"size": 16}, fmt='.1f')  # font size
            plt.show()
            name_suffix = '_ctrlcorrected' if check_zeros else ''
            add_ctrl = '.ctrl' if ctrl_flag else ''
            plt.savefig(f"Results_slice_{ds_info['slice_len']}.{train_config['model_postfix']}{add_ctrl}{name_suffix}.pdf")
            plt.clf()
            print('-------------------------------------------')
            print('-------------------------------------------')
            print('Global confusion matrix (%) (all trials)')
            print(cm)

            #plt.figure(figsize=(30, 1))
            #plt.imshow(tr[:1000, :].T)
            #plt.show()
        else:
            ###################### TESTING WITH VALIDATION DATA #########################
            print(f'Test Analysis for model {train_config["model_postfix"]} with slice {ds_info["slice_len"]}:')
            # Num. params
            total_params = sum(p.numel() for p in model.parameters())
            print(f'TOTAL params        {total_params}')

            test_dataloader = DataLoader(ds_test, batch_size=train_config['batch_size'], shuffle=False)

            size = len(test_dataloader.dataset)
            correct = 0
            conf_matrix = np.zeros((train_config['Nclass'], train_config['Nclass']))
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(device)
                    pred = model(X)
                    correct += (pred.cpu().argmax(1) == y).type(torch.float).sum().item()
                    conf_matrix += conf_mat(y, pred.cpu().argmax(1), labels=list(range(train_config['Nclass'])))
            correct /= size
            # Accuracy
            print(
                f"Test Error: \n "
                f"Accuracy: {(100 * correct):>0.2f}%"
            )
            # Conf. Matrix
            conf_matrix = conf_matrix.astype('float')
            for r in range(conf_matrix.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(conf_matrix[r, :])
                conf_matrix[r, :] = conf_matrix[r, :] / sum_row  * 100. # compute in percentage
            axis_lbl = ['eMBB', 'mMTC', 'URLLc'] if conf_matrix.shape[0] == 3 else ['eMBB', 'mMTC', 'URLLc', 'ctrl']
            df_cm = pd.DataFrame(conf_matrix, axis_lbl, axis_lbl)
            # plt.figure(figsize=(10,7))
            sn.set(font_scale=1.8)  # for label size
            sn.heatmap(df_cm, vmin=0, vmax=100, annot=True, cmap=sn.color_palette("light:b", as_cmap=True), annot_kws={"size": 25}, fmt='.1f')  # font size
            plt.show()
            add_ctrl = '.ctrl' if ctrl_flag else ''
            name_suffix = '_ctrlcorrected' if args.ds_file.split('_')[-2] == 'ctrlcorrected' else ''
            plt.savefig(f"Results_slice_{ds_info['slice_len']}.{train_config['model_postfix']}{add_ctrl}{name_suffix}_test.pdf")
            plt.clf()
            print('-------------------------------------------')
            print('Global confusion matrix (%) (validation split)')
            print(conf_matrix)
            # Inference time analysis
            inputs, _ = next(iter(test_dataloader))  
            sample_input = torch.unsqueeze(inputs[0], 0)
            sample_input = sample_input.to(device)
            m, sd = timing_inference_GPU(sample_input, model)
            print('-------------------------------------------')
            print('Inference time analysis:')
            print(f'Mean: {m}, Standard deviation: {sd}')









