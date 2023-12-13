import logging
import numpy as np
from typing import Dict
from torch import nn
from torch.utils.data import DataLoader
from python.ORAN_dataset import *
from python.ORAN_models import *
from python.visual_xapp_inference import process_norm_params
#from vit_pytorch import ViT
import time
from xapp_control import *


def main(model_type, torch_model_path, norm_param_path, Nclass, all_feats_raw=31):

    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    control_sck = open_control_socket(4200)

    pos_enc = False # not supported at the moment

    colsparams, indexes_to_keep, map_feat2KPI, num_feats, slice_len = process_norm_params(all_feats_raw,
                                                                                          norm_param_path)

    # initialize the KPI matrix
    kpi = []
    last_timestamp = 0
    curr_timestamp = 0

    # initialize the ML model
    print('Init ML model...')
    if model_type in [TransformerNN, TransformerNN_v2]:
        model = model_type(classes=Nclass, slice_len=slice_len, num_feats=num_feats, use_pos=pos_enc, nhead=1,
                             custom_enc=True)
    elif model_type == ConvNN:
        model = model_type(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
    else:
        # TODO
        print("ViT/other model is not yet supported. Aborting.")
        exit(-1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(torch_model_path, map_location='cuda:0')['model_state_dict'])
    else:
        device = 'cpu'
        model.load_state_dict(torch.load(torch_model_path, map_location='cpu')['model_state_dict'])
    model.to(device)
    model.eval()
    rand_x = torch.Tensor(np.random.random((1, slice_len, num_feats)))
    rand_x = rand_x.to(device)
    pred = model(rand_x)
    print('Dummy predict', pred)
    print('Start listening on E2 interface...')

    count_pkl = 0
    cont_data_sck = ""
    isCont = False
    while True:
        data_sck = receive_from_socket(control_sck)
        if len(data_sck) <= 0:
            if len(data_sck) == 0:
                #logging.info('Socket received 0')
                continue
            else:
                logging.info('Negative value for socket')
                break
        else:
            logging.info('Received data: ' + repr(data_sck))
            # with open('/home/kpi_new_log.txt', 'a') as f:
            #     f.write('{}\n'.format(data_sck))


            data_sck = data_sck.replace(',,', ',')
            data_sck_m = ''

            if data_sck[0] == 'm':
                # we need to recive more and piece together the whole message
                while data_sck[0] == 'm':
                    data_sck_m = data_sck_m + data_sck[1:]

                    # get more data
                    data_sck = receive_from_socket(control_sck)
                    if len(data_sck) <= 0:
                        if len(data_sck) == 0:
                            # logging.info('Socket received 0')
                            continue
                        else:
                            logging.info('Negative value for socket')
                            break
                    else:
                        logging.info('Received data: ' + repr(data_sck))
                        data_sck = data_sck.replace(',,', ',')

                # now we have to get the final message without an m
                data_sck = receive_from_socket(control_sck)
                if len(data_sck) <= 0:
                    if len(data_sck) == 0:
                        # logging.info('Socket received 0')
                        continue
                    else:
                        logging.info('Negative value for socket')
                        break
                else:
                    logging.info('Received data: ' + repr(data_sck))
                    data_sck = data_sck.replace(',,', ',')
                data_sck_m = data_sck_m + data_sck

                #finally rename for the rest of the program
                data_sck = data_sck_m

            kpi_new = np.fromstring(data_sck, sep=',')

            if kpi_new.shape[0] < all_feats_raw:
                logging.info('Discarding KPI: too short ')
                continue # discard incomplete KPIs

            assert kpi_new.shape[0] == all_feats_raw, "Check that we are indeed working with the intended number of raw KPIs"
            # check to see if the recently received KPI is actually new
            curr_timestamp = kpi_new[0] # first feature is timestamp

            if curr_timestamp > last_timestamp:
                kpi_raw = kpi_new.copy()
                # let's remove the KPIs we don't need
                kpi_filt = kpi_new[indexes_to_keep]
                last_timestamp = curr_timestamp
                if len(kpi) < slice_len:
                    # if the incoming KPI list is empty, just add the incoming KPIs
                    kpi.append(kpi_filt)
                else:
                    # to insert, we pop the first element of the list
                    kpi.pop(0)
                    # and append the last incoming KPI set
                    kpi.append(kpi_filt)
                    # here we have the new input ready for the ML model
                    # let's create a numpy array
                    np_kpi = np.array(kpi)
                    # let's normalize each columns based on the params derived while training
                    assert (np_kpi.shape[1] == len(list(map_feat2KPI.keys()))), "Check that filtered features has same size of mapped KPIs for normalization"
                    for f in range(np_kpi.shape[1]):
                        c = map_feat2KPI[f]
                        print('***** ' + colsparams[c]['name']+' ('+str(c)+') *****')
                        if np.any(np_kpi[:, f] > colsparams[c]['max']) or np.any(np_kpi[:, f] < colsparams[c]['min']):
                            print("Clipping ", colsparams[c]['min'], "< x <", colsparams[c]['max'])
                            np_kpi[:, f] = np.clip(np_kpi[:, f], colsparams[c]['min'], colsparams[c]['max'])

                        print('Un-normalized vector'+repr(np_kpi[:, f]))
                        np_kpi[:, f] = (np_kpi[:, f] - colsparams[c]['min']) / (
                                    colsparams[c]['max'] - colsparams[c]['min'])
                        print('Normalized vector: '+repr(np_kpi[:, f]))
                    # and then pass it to our model as a torch tensor
                    t_kpi = torch.Tensor(np.expand_dims(np_kpi, axis=0)).to(device)
                    try:
                        pred = model(t_kpi)
                        this_class = pred.argmax(1)
                        print('Predicted class ' + str(pred.argmax(1)))
                        pickle.dump({'input': np_kpi, 'label': this_class, 'input_raw': kpi_raw}, open('/home/class_output__'+str(int(time.time()*1e3))+'.pkl', 'wb'))
                        count_pkl += 1
                    except:
                        print('ERROR while predicting class')

                    # with open('/home/kpi_log.txt', 'a') as f:
                    #   f.write(str(np_kpi[:, :5]) + '\n')




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to TRACTOR model to load."  )
    parser.add_argument("--norm_param_path", required=True, default="", help="Normalization parameters path.")
    parser.add_argument("--model_type", default="Tv1", choices=['CNN', 'Tv1', 'Tv2', 'ViT'], help="Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)")
    args, _ = parser.parse_known_args()

    if args.model_type is not None:
        if args.model_type == 'Tv1':
            model_type = TransformerNN
        elif args.model_type == 'Tv2':
            model_type = TransformerNN_v2
        elif args.model_type == 'ViT':
            #transformer = ViT
            print("Transformer type "+args.transformer+" is not yet supported.")
            exit(-1)
        elif args.model_type == 'CNN':
            model_type = ConvNN

    torch_model_path = args.model_path
    norm_param_path = args.norm_param_path

    Nclass = 4

    main(model_type, torch_model_path, norm_param_path, Nclass)
