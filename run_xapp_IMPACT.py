import logging
import numpy as np
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from python.ORAN_dataset import *
from python.ORAN_models import ConvNN as global_model
import time
from xapp_control import *


class NeuModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


# Function to normalize data
def normalize(data):
    temp = []
    for moment in data:
        for i in range(len(moment)):
            moment[i] /= maximum_metrics[i]
        temp.append(moment)
    return np.array(temp).flatten()


# Function to make predictions
def predict_newdata(model, input_data):
    if not torch.is_tensor(input_data):
        input_data = torch.tensor(input_data, dtype=torch.float)
        #print(input_data)
    pred = model(input_data)
    val, predicted = torch.max(pred,0)
    predicted_label = predicted.item()
    # if predicted_label == 0:
    #     predicted_label = "No Interference"
    # elif predicted_label == 1:
    #     predicted_label = "Interference"
    # else:
    #     predicted_label = "Control"
    return predicted_label


maximum_metrics = {0: 16.0, 1: 15.0193, 2: 39.6237, 3: 87.5}
BLOCK_SIZE = 5


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

    slice_len = 32
    Nclass = 4
    num_feats = 17
    torch_model_path = 'model/model_weights__slice32.pt'
    norm_param_path = 'model/cols_maxmin.pkl'
    colsparam_dict = pickle.load(open(norm_param_path, 'rb'))
    # initialize the KPI matrix (4 samples, 19 KPIs each)
    #kpi = np.zeros([slice_len, num_feats])
    kpi = []
    kpi_i = []
    last_timestamp = 0
    curr_timestamp = 0

    # initialize the ML model
    print('Init ML models...')
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
    print('Dummy slice prediction', pred)

    model_i = torch.load('model/best_model_pytorch.pt')
    rand_i = np.random.random(20)
    pred_i = predict_newdata(model_i, rand_i)
    print('Dummy interference prediction', pred_i)

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
            if kpi_new.shape[0] < 31:
                logging.info('Discarding KPI: too short ')
                continue # discard incomplete KPIs
                        # [TODO] this is to address the multiple 'm' case, but not ideal like this

            # check to see if the recently received KPI is actually new
            # kpi_process = kpi_new[np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]

            if kpi_new[2] == 1010123456002:
                curr_timestamp = kpi_new[0]
                # let's remove the KPIs we don't need
                kpi_filt = kpi_new[np.array([9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]
                # interference needs [16, 19, 23, 21]
                kpi_filt_i = kpi_new[np.array([16, 19, 23, 21])]

                if curr_timestamp > last_timestamp:
                    last_timestamp = curr_timestamp

                    # first do traffic class prediction
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
                            # pickle.dump((np_kpi, this_class),
                            #             open('/home/class_output__'+str(int(time.time()*1e3))+'.pkl', 'wb'))
                            count_pkl += 1
                        except:
                            logging.info('ERROR while predicting class')

                    # then do interference prediction
                    if len(kpi_i) < BLOCK_SIZE:
                        kpi_i.append(kpi_filt_i)
                    else:
                        kpi_i.pop(0)
                        kpi_i.append(kpi_filt_i)
                        np_kpi_i = np.array(kpi_i)
                        np_kpi_i = normalize(np_kpi_i)
                        output = predict_newdata(model_i, np_kpi_i)
                        logging.info('Predicted interference ' + str(output))
                        # pickle.dump((np_kpi_i, output),
                        #            open('/home/interference_output__' + str(int(time.time() * 1e3)) + '.pkl', 'wb'))

                    # TODO: Add control messages to this logic
                    if this_class == 0:
                        if output == 0:
                            print('embb no interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('embb no interference \n')
                        if output == 1:
                            print('embb with interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('embb with interference \n')
                        if output == 2:
                            print('unexpected result: embb and cntrl')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('unexpected result: embb and cntrl \n')
                    if this_class == 1:
                        if output == 0:
                            print('mmtc no interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('mmtc no interference \n')
                        if output == 1:
                            print('mmtc with interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('mmtc with interference \n')
                        if output == 2:
                            print('unexpected result: mmtc and cntrl')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('unexpected result: mmtc and cntrl \n')
                    if this_class == 2:
                        if output == 0:
                            print('urllc no interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('urllc no interference \n')
                        if output == 1:
                            print('urllc with interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('urllc with interference \n')
                        if output == 2:
                            print('unexpected result: urllc and cntrl')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('unexpected result: urllc and cntrl \n')
                    if this_class == 3:
                        if output == 0:
                            print('unexpected result: cntrl no interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('unexpected result: cntrl no interference \n')
                        if output == 1:
                            print('unexpected result: cntrl with interference')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('unexpected result: cntrl with interference \n')
                        if output == 2:
                            print('cntrl')
                            with open('/home/kpi_log.txt', 'a') as f:
                                f.write('cntrl \n')




if __name__ == '__main__':
    main()
