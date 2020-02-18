import pickle
from matplotlib import pyplot as plt

def main():
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)
    plt.plot([i['acc'] for i in history])
    plt.savefig('accuracy.png')
    plt.close()
    plt.plot([i['loss'] for i in history])
    plt.savefig('loss.png')


if __name__ == "__main__":
    main()