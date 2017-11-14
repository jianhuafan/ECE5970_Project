import rnn_model
import argparse
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str, default='lstm',
                        help='tpye of rnn:lstm, gru or rnn')
    parser.add_argument('--load_model', type = bool, default = True,
                        help='whether load a pre-traind model')
    parser.add_argument('--weights_fpath',type=str, default = 'model/',
                        help='path of weitght')
    parser.add_argument('--num_hidden', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')  
    parser.add_argument('--batch_size', type=int, default=10,
                       help='minibatch size')
    parser.add_argument('--train_seq_length', type=int, default=15,
                       help='RNN sequence length in training phase')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='number of epochs')
    parser.add_argument('--grad_clipping', type=float, default=1.,
                       help='clip gradients at this value') 
    parser.add_argument('--lr', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97,
                       help='the decay rate of learning rate')
    parser.add_argument('--lr_decay_after', type=int, default =10,
                       help='number of epochs to start decaying the learning rate') 
    parser.add_argument('--input_train_fpath', type=str, default='data/input_train.csv',
                       help='data directory containing input_train')
    parser.add_argument('--output_train_fpath', type=str, default='data/output_train.csv',
                       help='data directory containing output_train')
    parser.add_argument('--input_test_fpath', type=str, default='data/input_test.csv',
                       help='data directory containing input_test')
    parser.add_argument('--output_test_fpath', type=str, default='data/output_test.csv',
                       help='data directory containing output_test')
    args = parser.parse_args()
    test(args)

def test(args):
    model = args.model # type of RNN
    weights_fpath = args.weights_fpath  # weights will be stored here
    input_train_fpath = args.input_train_fpath
    output_train_fpath = args.output_train_fpath
    input_test_fpath = args.input_test_fpath
    output_test_fpath = args.output_test_fpath
    max_epochs = args.max_epochs
    lr = args.lr
    lr_decay = args.lr_decay
    lr_decay_after = args.lr_decay_after
    grad_clipping = args.grad_clipping  
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    batch_size = args.batch_size
    train_seq_length = args.train_seq_length
    load_model = args.load_model

    print("rnn: {}".format(model))
    print("num_hidden: {}".format(num_hidden))
    print("max_epochs: {}".format(max_epochs))

    print('Start Testing')

    g = rnn_model.build_graph(model, lr, lr_decay, lr_decay_after,
        grad_clipping, num_hidden, num_layers, batch_size, train_seq_length)
    rnn_model.test_graph(load_model, model, num_layers, num_hidden, g, batch_size,
        train_seq_length, max_epochs, weights_fpath, input_train_fpath,
        input_test_fpath, output_train_fpath, output_test_fpath)




if __name__ == '__main__':
    main()