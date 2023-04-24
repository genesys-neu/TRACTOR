import logging
import numpy as np
from typing import Dict
from torch import nn
from torch.utils.data import DataLoader
from python.ORAN_dataset import *
from python.torch_train_ORAN_colosseum import ConvNN as global_model
import time
from xapp_control import *


def main():
    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    control_sck = open_control_socket(4200)

    slice_len = 8
    Nclass = 4
    num_feats = 17
    torch_model_path = 'model/model_weights__slice8.pt'
    norm_param_path = 'logs/cols_maxmin.pkl'
    colsparam_dict = pickle.load(open(norm_param_path, 'rb'))
    # initialize the KPI matrix (4 samples, 19 KPIs each)
    #kpi = np.zeros([slice_len, num_feats])
    kpi = []
    last_timestamp = 0
    curr_timestamp = 0

    # initialize the ML model
    print('Init ML model...')
    model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(torch_model_path, map_location='cuda:0'))
    else:
        device = 'cpu'
        model.load_state_dict(torch.load(torch_model_path))
    model.to(device)
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

            """
            if data_sck[0] == 'm':
                print('Multiple recv')  # TODO handle this case
                if not isCont:
                    isCont = True   # activate continue mode
                    cont_data_sck = data_sck[1:]    # init the string
                else:    # we are already in continue mode
                    cont_data_sck += data_sck[1:]
                continue    # don't process the string and continue appending
            elif isCont:
                cont_data_sck += data_sck
                data_sck = cont_data_sck
                cont_data_sck = ""
                isCont = False
            """


            data_sck = data_sck.replace(',,', ',')

            if data_sck[0] == 'm':
                data_sck = data_sck[1:]

            kpi_new = np.fromstring(data_sck, sep=',')
            if kpi_new.shape[0] < 31:
                continue # discard incomplete KPIs
                        # [TODO] this is to address the multiple 'm' case, but not ideal like this

            # check to see if the recently received KPI is actually new
            # kpi_process = kpi_new[np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]
            curr_timestamp = kpi_new[0]
            # let's remove the KPIs we don't need
            kpi_filt = kpi_new[np.array([9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]

            if curr_timestamp > last_timestamp:
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
                    assert (np_kpi.shape[1] == len(list(colsparam_dict.keys())))
                    for c in range(np_kpi.shape[1]):
                        print('*****', c, '*****')
                        logging.info('Un-normalized vector'+repr(np_kpi[:, c]))
                        np_kpi[:, c] = (np_kpi[:, c] - colsparam_dict[c]['min']) / (
                                    colsparam_dict[c]['max'] - colsparam_dict[c]['min'])
                        logging.info('Normalized vector: '+repr(np_kpi[:, c]))
                    # and then pass it to our model as a torch tensor
                    t_kpi = torch.Tensor(np_kpi.reshape(1, np_kpi.shape[0], np_kpi.shape[1])).to(device)
                    try:
                        pred = model(t_kpi)
                        this_class = pred.argmax(1)
                        logging.info('Predicted class ' + str(pred.argmax(1)))
                        pickle.dump((np_kpi, this_class), open('/home/class_output__'+str(int(time.time()*1e3))+'.pkl', 'wb'))
                        count_pkl += 1
                    except:
                        logging.info('ERROR while predicting class')

                    # with open('/home/kpi_log.txt', 'a') as f:
                    #   f.write(str(np_kpi[:, :5]) + '\n')



if __name__ == '__main__':
    main()
