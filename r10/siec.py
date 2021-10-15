#!/usr/bin/env python3
## Python
from math import floor

## Archiwzacja
import os
import argparse

## Moduł uczenia głębokiego
import mxnet as mx

## Przetwarzanie danych
import numpy as np
import pandas as pd


## Parametry, których nie będziemy konfigurować przy użyciu konsoli
DATA_SEGMENTS = { 'tr': 0.6, 'va': 0.2, 'tst': 0.2}
THRESHOLD_EPOCHS = 5
COR_THRESHOLD = 0.0005

## Konfiguracja parsera standardowego wejścia
parser = argparse.ArgumentParser()

## Kształt danych
parser.add_argument('--win',        type=int,   default=24*7)
parser.add_argument('--h',          type=int,   default=3)

## Specyfikacja modelu
parser.add_argument('--model',      type=str,   default='rnn_model')
## Komponenty CNN
parser.add_argument('--sz-filt',    type=int,   default=8)
parser.add_argument('--n-filt',     type=int,   default=10)
## Komponenty RNN 
parser.add_argument('--rnn-units',  type=int,   default=10)

## Szczegóły procesu uczenia
parser.add_argument('--batch-n',    type=int,   default=1024)
parser.add_argument('--lr',         type=float, default=0.0001)
parser.add_argument('--drop',       type=float, default=0.2)
parser.add_argument('--n-epochs',   type=int,   default=30)

## Archiwizacja 

parser.add_argument('--data-dir',   type=str,   default='../data')
parser.add_argument('--save-dir',   type=str,   default=None)





def prepared_data(data_dir, win, h, model_name):
    df = pd.read_csv(os.path.join(data_dir, 'electricity.diff.txt'),
                     sep=',', header=0)
    x = df.values
    ## Normalizacja danych. Zauważ że, ponieważ wykonujemy ją w oparciu o parametry
    ## mierzone dla całego szeregu, wprowadza ona zjawisko lookahead
    
    ## W przypadku bardziej złożonego procesu przetwarzania
    ## zastosowalibyśmy statystyki ruchome, aby uniknąć tego problemu
    x = (x - np.mean(x, axis = 0)) / (np.std(x, axis = 0))
    
    if model_name == 'fc_model': ## Format NC
        ### Odwołania do pierwszego i drugiego kroku w przeszłość umieszczamy w postaci pojedynczego, płaskiego wejścia
        X = np.hstack([x[1:-h], x[0:-(h+1)]])
        Y = x[(h+1):]
        return (X, Y)
    else:                        ## Format TNC
        # Prealokacja zmiennych X i Y
        # rozmiar X = liczba przykładów * szerokość okna czasowego * liczba kanałów (NTC)
        X = np.zeros((x.shape[0] - win - h, win, x.shape[1]))
        Y = np.zeros((x.shape[0] - win - h, x.shape[1]))
        for i in range(win, x.shape[0] - h):
            ## Cel/poszukiwana wartość znajdują się o h kroków do przodu
            Y[i-win] = x[i + h - 1 , :]
            ## Dane wejściowe to ostatnie win kroków
            X[i-win] = x[(i - win) : i , :]
        
        return (X, Y)
        
def train(symbol, iter_train, valid_iter, iter_test, data_names, label_names, save_dir):
    ## Zapis informacji o procesie uczenia/ wyników 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    printFile = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    def print_to_file(msg):
        print(msg)
        print(msg, file = printFile, flush = True)
    ## Nagłówek charakteryzujący zapisywane informacje
    print_to_file('Epoka   Korelacja zbiór uczący   Korelacja zbiór walidacyjny')
    
    ## Zapamiętanie wartości z porzedniej epoki, które wykorzystamy
    ## do ustalenia progowej wartości dla, której następuje poprawa;
    ## jeżeli poprawa zachodzi zbyt wolno, przerywamy uczenie
    buf = RingBuffer(THRESHOLD_EPOCHS)
    old_val = None
    
    
    ## Trochę standardowego kodu potrzebnego do obsługi biblioteki mxnet
    ## Domyślne mamy jedno gpu o indeksie 0, w przypadku posiadania karty graficznej firmy innej niż NVIDIA
        ## należy mx.gpu(0) zastąpić mx.cpu(0)
    
    devs=[mx.gpu(0)]
    module = mx.mod.Module(symbol,
                           data_names=data_names,
                           label_names=label_names,
                           context=devs)
    module.bind(data_shapes=iter_train.provide_data,
                label_shapes=iter_train.provide_label)
    module.init_params(mx.initializer.Uniform(0.1))
    module.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': args.lr})
    
    ## Proces uczenia
    for epoch in range( args.n_epochs):
        iter_train.reset()
        iter_val.reset()
        for batch in iter_train:
            # Obliczanie predykcji
            module.forward(batch, is_train=True)
            # Obliczanie gradientu
            module.backward()
            # Aktualizacja parametrów
            module.update()
   
        ## Rezultaty procesu uczenia
        train_pred = module.predict(iter_train).asnumpy()
        train_label = iter_train.label[0][1].asnumpy()
        train_perf = evaluate_and_write(train_pred, train_label,
                                     save_dir, 'train', epoch)
        ## Rezultaty walidacji
        val_pred = module.predict(iter_val).asnumpy()
        val_label = iter_val.label[0][1].asnumpy()
        val_perf = evaluate_and_write(val_pred, val_label, 
                                   save_dir, 'valid', epoch)
        print_to_file('%d           %f                 %f ' %
                     (epoch, train_perf['COR'], val_perf['COR']))

        # Jeżeli nie mamy wystarczającej liczby pomiarów, przechodzimy dalej
        if epoch > 0:
            buf.append(val_perf['COR'] - old_val)
        # Jeżeli zaszła jakaś zmiana, sprawdzamy ją
        if epoch > 2:
            vals = buf.get()
            vals = [v for v in vals if v != 0]
            if sum([v < COR_THRESHOLD for v in vals]) == len(vals):
                print_to_file('Wcześniejsze wyjście')
                break
        old_val = val_perf['COR']

    ## Testowanie
    test_pred = module.predict(iter_test).asnumpy()
    test_label = iter_test.label[0][1].asnumpy()
    test_perf = evaluate_and_write(test_pred, test_label,
    save_dir, 'tst', epoch)
    print_to_file('WYDAJNOŚĆ NA ZBIORZE TESTOWYM')
    print_to_file(test_perf)

def evaluate_and_write(pred, label, save_dir, mode, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pred_df = pd.DataFrame(pred)
    label_df = pd.DataFrame(label)
    pred_df.to_csv( os.path.join(save_dir, '%s_pred%d.csv'
                                 % (mode, epoch)))
    label_df.to_csv(os.path.join(save_dir, '%s_label%d.csv'
                                 % (mode, epoch)))
    return { 'COR': COR(label,pred) }

def COR(label, pred):
    label_demeaned = label - label.mean(0)
    label_sumsquares = np.sum(np.square(label_demeaned), 0)
    
    pred_demeaned = pred - pred.mean(0)
    pred_sumsquares = np.sum(np.square(pred_demeaned), 0)
    
    cor_coef = np.diagonal(np.dot(label_demeaned.T, pred_demeaned))/np.sqrt(label_sumsquares * pred_sumsquares)
    return np.nanmean(cor_coef)

def print_to_file(msg):
    print(msg, file = printFile, flush = True)
    print_to_file(args)


def fc_model(iter_train, window, filter_size, num_filter, dropout):
    X = mx.sym.Variable(iter_train.provide_data[0].name)
    Y = mx.sym.Variable(iter_train.provide_label[0].name)
    
    output = mx.sym.FullyConnected(data=X, num_hidden = 20)
    output = mx.sym.Activation(output, act_type = 'relu')
    output = mx.sym.FullyConnected(data=output, num_hidden = 10)
    output = mx.sym.Activation(output, act_type = 'relu')
    output = mx.sym.FullyConnected(data = output, num_hidden = 321)
    
    loss_grad = mx.sym.LinearRegressionOutput(data = output,
                                              label = Y)
    return loss_grad, [v.name for v in iter_train.provide_data], [v.name for v in iter_train.provide_label]

def cnn_model(iter_train, input_feature_shape, X, Y,
              win, sz_filt, n_filter, drop):
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, win, -1))
    ## Konwolucja oczekuje wejścia 4d (N x liczba kanałów x wysokość x szerokość)
    ## W naszym przypadku liczba kanałów = 1 (tak jak w przypadku czarno-białego obrazu)
    ## wysokość = czas, szerokość = liczba kanałów/liczba lokalizacji pomiarowych

    cnn_output = mx.sym.Convolution(data=conv_input, 
                                    kernel=(sz_filt,
                                            input_feature_shape[2]),
                                    num_filter=n_filter)
    cnn_output = mx.sym.Activation(data=cnn_output, act_type='relu')
    cnn_output = mx.sym.reshape(mx.sym.transpose(data=cnn_output,
                                                 axes=(0, 2, 1, 3)),
                                shape=(0, 0, 0))

    cnn_output = mx.sym.Dropout(cnn_output, p=drop)

    output = mx.sym.FullyConnected(data=cnn_output,
                                   num_hidden=input_feature_shape[2])
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad, [v.name for v in iter_train.provide_data], [v.name for v in iter_train.provide_label])

## Kod ten został stworzony do pracy z wagami wyeksportowanymi
## z TensorFlow, ale można go łatwo przystosować do pracy także z innymi
## https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnGRU

def calc_gru(X, weights, num_inputs, num_features):
    Us = weights[:(3*num_features*num_inputs)]
    Us = np.reshape(Us, [3, num_features, num_inputs])
    Ws = weights[(3*num_features*num_inputs):(3*num_features*num_features + 3*num_features*num_inputs)]
    Ws = np.reshape(Ws, [3, num_features, num_features])
    
    Bs = weights[(-6 * num_features) :]
    Bs = np.reshape(Bs, [6, num_features])
    s = np.zeros([129, num_features])
    h = np.zeros([129, num_features])
    for t in range(X.shape[0]):
        z = sigmoid(np.matmul(Us[0, :, :], X[t, :]) +
                    np.matmul(Ws[0, :, :], s[t, :]) + Bs[0, :] + Bs[3, :])
        r = sigmoid(np.matmul(Us[1, :, :], X[t, :]) +
                    np.matmul(Ws[1, :, :], s[t, :]) + Bs[1, :] + Bs[4, :])
        h[t+1, :] = np.tanh(np.matmul(Us[2, :, :], X[t, :]) +
                            Bs[2, :] +
                            r*(np.matmul(Ws[2, :, :], s[t, :]) + Bs[5, :]))
        s[t+1, :] = (1 - z)*h[t + 1, :] + z*s[t, :]
    return h, s

def rnn_model(iter_train, window, filter_size, num_filter, dropout):
    input_feature_shape = iter_train.provide_data[0][1] 
    X = mx.sym.Variable(iter_train.provide_data[0].name) 
    Y = mx.sym.Variable(iter_train.provide_label[0].name)
    
    rnn_cells = mx.rnn.SequentialRNNCell() 
    rnn_cells.add(mx.rnn.GRUCell(num_hidden=args.rnn_units)) 
    rnn_cells.add(mx.rnn.DropoutCell(dropout))
    outputs, _ = rnn_cells.unroll(length=window, inputs=X,
                                  merge_outputs=False)
    output = mx.sym.FullyConnected(data=outputs[-1],
                                   num_hidden = input_feature_shape[2])
    loss_grad = mx.sym.LinearRegressionOutput(data = output, 
                                              label = Y)
    return loss_grad, [v.name for v in iter_train.provide_data], [v.name for v in iter_train.provide_label]

def simple_lstnet_model(iter_train, input_feature_shape, X, Y,
                        win, sz_filt, n_filter, drop):
    
    ## Aby skorzystać z paddingu wejście musi być cztero- lub pięciowymiarowe
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, win, -1)) 

    ## Komponent konwolucyjny
    ## Dodajemy margines na końcach okna 
    cnn_output = mx.sym.pad(data=conv_input,
                        mode="constant",
                        constant_value=0,
                        pad_width=(0, 0,
                                   0, 0,
                                   0, sz_filt - 1,
                                   0, 0)) 
    cnn_output = mx.sym.Convolution(data=cnn_output,
                                    kernel=(sz_filt,
                                            input_feature_shape[2]),
                                    num_filter=n_filter)
    cnn_output = mx.sym.Activation(data=cnn_output,
                                   act_type='relu')
    cnn_output = mx.sym.reshape(mx.sym.transpose(data=cnn_output,
                                             axes=(0, 2, 1, 3)),
                                shape = (0, 0, 0))
    cnn_output = mx.sym.Dropout(cnn_output, p=drop)
    
    ## Komponent rekurencyjny
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    stacked_rnn_cells.add(mx.rnn.GRUCell(num_hidden=args.rnn_units))
    outputs, _ = stacked_rnn_cells.unroll(length=win,
                                          inputs=cnn_output,
                                          merge_outputs=False)
    rnn_output = outputs[-1]
    n_outputs = input_feature_shape[2]
    cnn_rnn_model = mx.sym.FullyConnected(data=rnn_output,
                                          num_hidden=n_outputs)
    ## Komponent AR
    ar_outputs = []
    for i in list(range(input_feature_shape[2])):
        ar_series = mx.sym.slice_axis(data=X,
                                      axis=2,
                                      begin=i,
                                      end=i+1)
        fc_ar = mx.sym.FullyConnected(data=ar_series, num_hidden=1)
        ar_outputs.append(fc_ar)
    ar_model = mx.sym.concat(*ar_outputs, dim=1)
    
    output = cnn_rnn_model + ar_model
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad,[v.name for v in iter_train.provide_data], [v.name for v in iter_train.provide_label])

##################################
## Przygotowanie danych wejściowych ##
##################################


def prepare_iters(data_dir, win, h, model, batch_n):
    X, Y = prepared_data(data_dir, win, h, model)

    n_tr = int(Y.shape[0] * DATA_SEGMENTS['tr'])
    n_va = int(Y.shape[0] * DATA_SEGMENTS['va'])

    X_tr, X_valid, X_test = X[                      : n_tr], \
                               X[n_tr             : n_tr + n_va], \
                               X[n_tr + n_va : ]
    Y_tr, Y_valid, Y_test = Y[                      : n_tr], \
                               Y[n_tr             : n_tr + n_va], \
                               Y[n_tr + n_va : ]
    
    iter_tr = mx.io.NDArrayIter(data       = X_tr,
                                label      = Y_tr,
                                batch_size = batch_n)
    iter_val = mx.io.NDArrayIter(data       = X_valid,
                                 label      = Y_valid,
                                 batch_size = batch_n)
    iter_test = mx.io.NDArrayIter(data       = X_test,
                                  label      = Y_test,
                                  batch_size = batch_n)

    return (iter_tr, iter_val, iter_test)
    
## Dzięki uprzejmości https://www.saltycrane.com/blog/2007/11/python-circular-buffer/
class RingBuffer:
    def __init__(self, size):
        self.data = [0 for i in range(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

if __name__ == '__main__':
    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args()
    # Tworzenie iteratorów
    iter_train, iter_val, iter_test = prepare_iters(
        args.data_dir, args.win, args.h,
        args.model, args.batch_n)
    
    ## Przygotowywanie odpowiednich symboli
    input_feature_shape = iter_train.provide_data[0][1]
    X = mx.sym.Variable(iter_train.provide_data[0].name ) 
    Y = mx.sym.Variable(iter_train.provide_label[0].name)
    
    # Przygotowanie modelu
    model_dict = { 'fc_model' : fc_model,
                  'rnn_model' : rnn_model,
                  'cnn_model' : cnn_model,
                  'simple_lstnet_model' : simple_lstnet_model
                 }
    model = model_dict[args.model]
    if args.model=='cnn_model' or args.model=='simple_lstnet_model':
        symbol, data_names, label_names = model(iter_train,
                                                input_feature_shape, X, Y,
                                                args.win, args.sz_filt,
                                                args.n_filt, args.drop)
        
    else:
        symbol, data_names, label_names = model(iter_train,
                                                args.win, args.sz_filt,
                                                args.n_filt, args.drop)
        

    
    ## Proces uczenia
    train(symbol, iter_train, iter_val, iter_test,
          data_names, label_names, args.save_dir)

