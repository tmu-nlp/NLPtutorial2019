import dill
import train_sr as sr

def main():
    with open('train_file', 'rb') as f:
        W = dill.loads(f.read())

    # *data, = sr.load_data('../../data/mstparser-en-test.dep')
    data = sr.load_data('../../data/mstparser-en-train.dep')
    # print(data)
    with open('out.txt', 'w') as f:
        for sentence in data:
            heads = sr.shift_reduce(sentence, W, mode='test')
            for token, head in zip(sentence, heads[1:]):
                token.attrs[6] = token.head
                print('\t'.join(map(str,token.attrs)), file=f)
            print(file=f)

if __name__ == '__main__':
    main()
