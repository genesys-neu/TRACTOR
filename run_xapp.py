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

    num_slices = 8
    Nclass = 4
    num_feats = 17
    torch_model_path = 'model/model_weights__slice8.pt'
    # initialize the KPI matrix (4 samples, 19 KPIs each)
    kpi = np.empty([num_slices, num_feats])

    # initialize the ML model
    print('Init ML model...')
    model = global_model(classes=Nclass, slice_len=num_slices, num_feats=num_feats)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(torch_model_path, map_location='cuda:0'))
    else:
        device = 'cpu'
        model.load_state_dict(torch.load(torch_model_path))
    model.to(device)
    rand_x = torch.Tensor(np.random.random((1, num_slices, num_feats)))
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
            if kpi_new[0] > kpi[num_slices-1, 0]:
                # roll all KPIs up one
                kpi = np.roll(kpi, -1, axis=0)
                # update the last row with the new KPIs
                kpi[num_slices-1, :] = kpi_new[np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]
                with open('/home/kpi_log.txt', 'a') as f:
                    f.write('{}\n'.format(kpi))


if __name__ == '__main__':
    main()
