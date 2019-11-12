from sklearn.model_selection import train_test_split
import pandas as pd
import sys

try:
    infile = sys.argv[1]
    X = pd.read_csv(infile, header=None)
except:
    try:
        X = pd.read_csv('../data/cmsa_full.csv', header=None)
    except:
        print('cmsa_full.csv not found')
        raise SystemExit
    else:
        print('loaded cmsa_full.csv')
else:
    print('loaded: {}'.format(infile))

# Split dataset 60/40
print('spliting dataset 60/40')
X_train, X_test = train_test_split(X, test_size=.4, stratify=X[0], random_state=0)

# Get 10% of training for dev
print('copying 10% of training set to dev')
X_dev = train_test_split(X_train, test_size=.1, stratify=X_train[0], random_state=0)[1]

X_train.to_csv('../data/train/cmsa_train.csv', index=False, header=False)
X_test.to_csv('../data/test/cmsa_test.csv', index=False, header=False)
X_dev.to_csv('../data/dev/cmsa_dev.csv', index=False, header=False)
print('done')
