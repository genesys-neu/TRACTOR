import logging
import numpy as np
from typing import Dict
from torch import nn
from torch.utils.data import DataLoader
from python.ORAN_dataset import *
from python.torch_train_ORAN_colosseum import ConvNN as global_model

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
    # initialize the KPI matrix (4 samples, 19 KPIs each)
    #kpi = np.zeros([slice_len, num_feats])
    kpi = []

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
            with open('/home/kpi_new_log.txt', 'a') as f:
                f.write('{}\n'.format(data_sck))

            kpi_new = np.fromstring(data_sck, sep=',')
            # check to see if the recently received KPI is actually new
            kpi_process = kpi_new[np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]
            # let's remove the KPIs we don't need
            kpi_filt = kpi_process[np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])]
            if len(kpi) == 0:
                # if the incoming KPI list is empty, just add the incoming KPIs
                kpi.append(kpi_filt)
            if 0 < len(kpi) < slice_len:
                # if we are still filling the buffer, let's make sure the new KPI has higher timestamp value
                # compared to the last set of KPIs recorded
                if kpi_process[0] > kpi[-1][0]:
                    kpi.append(kpi_filt)
            elif len(kpi) == slice_len:
                # again make sure we have a new set of KPIs by checking the timestamp
                if kpi_process[0] > kpi[-1][0]:
                    # to insert, we pop the first element of the list
                    kpi.pop(0)
                    # and append the last incoming KPI set
                    kpi.append(kpi_filt)
                    # here we have the new input ready for the ML model
                    with open('/home/kpi_log.txt', 'a') as f:
                        np_kpi = np.array(kpi)
                        f.write(str(np_kpi[:,:5])+'\n')


if __name__ == '__main__':
    main()