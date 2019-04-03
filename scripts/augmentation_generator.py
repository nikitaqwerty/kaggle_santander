import numpy as np

def generate_augment_class(X):
    while True:
        X_new = X.copy()
        ids = np.arange(X.shape[0])

        for c in range(X.shape[1]):
            np.random.shuffle(ids)
            X_new[:,c] = X[ids][:,c]
        
        for i in range(X.shape[0]):
            yield X_new[i]

def generate_augment(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    t_pos = 2
    t_neg = 1

    pos_count = (y == 1).sum() * t_pos
    neg_count = (y == 0).sum() * t_neg
    print pos_count, neg_count
    
    positive = generate_augment_class(X_pos)
    negative = generate_augment_class(X_neg)
    
    while True:
        y_balanced = np.array([1] * pos_count + [0] * neg_count)
        np.random.shuffle(y_balanced)
        
        for y_i in y_balanced:
            if y_i == 1:
                yield positive.next(), y_i
            elif y_i == 0:
                yield negative.next(), y_i
                
                
# Example / test
X1 = np.ones((10, 2)) * 10
X1[:, 1] = X1[:, 1] + np.arange(0, 10)
X2 = np.ones((100, 2))
X2[:, 1] = X2[:, 1] + np.arange(0, 1, 0.01)

y1 = np.ones((10,))
y2 = np.zeros((100,))

test_x = []
test_y = []
for i, (x, y) in enumerate(generate_augment(np.vstack([X1, X2]), np.concatenate([y1, y2]))):
    if i >= 240:
        break
        
    test_x.append(x)
    test_y.append(y)
    
test_x = np.row_stack(test_x)
test_y = np.array(test_y)
test_y.mean(), test_x.mean(axis=0)
