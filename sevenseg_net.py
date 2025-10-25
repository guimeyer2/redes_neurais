import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.5, seed=42):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(0, 1, size=(n_input, n_hidden)) * 0.1
        self.b1 = np.zeros((1, n_hidden))

        self.W2 = rng.normal(0, 1, size=(n_hidden, n_output)) * 0.1
        self.b2 = np.zeros((1, n_output))

        self.lr = lr

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = sigmoid(z2)
        return a1, z1, y_pred, z2

    def backward(self, X, y_true, a1, z1, y_pred, z2):
        N = X.shape[0]

        dE_dy = (2.0 / N) * (y_pred - y_true)
        dy_dz2 = sigmoid_deriv(z2)
        delta2 = dE_dy * dy_dz2

        dE_dW2 = a1.T @ delta2
        dE_db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(z1)
        dE_dW1 = X.T @ delta1
        dE_db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W2 -= self.lr * dE_dW2
        self.b2 -= self.lr * dE_db2
        self.W1 -= self.lr * dE_dW1
        self.b1 -= self.lr * dE_db1

    def train(self, X, y, epochs=20000, print_every=4000):
        for ep in range(1, epochs + 1):
            a1, z1, y_pred, z2 = self.forward(X)
            loss = mse(y, y_pred)
            self.backward(X, y, a1, z1, y_pred, z2)

            if ep % print_every == 0:
                print(f"[epoch {ep}] loss = {loss:.6f}")

    def predict(self, X):
        _, _, y_pred, _ = self.forward(X)
        return y_pred

def add_bitflip_noise(X, flip_prob=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noisy = X.copy()
    mask = rng.random(size=noisy.shape) < flip_prob
    noisy[mask] = 1 - noisy[mask]
    return noisy

if __name__ == "__main__":
    # Padrões dos 7 segmentos (a,b,c,d,e,f,g)
    segments_by_digit = {
        0: [1,1,1,1,1,1,0],
        1: [0,1,1,0,0,0,0],
        2: [1,1,0,1,1,0,1],
        3: [1,1,1,1,0,0,1],
        4: [0,1,1,0,0,1,1],
        5: [1,0,1,1,0,1,1],
        6: [1,0,1,1,1,1,1],
        7: [1,1,1,0,0,0,0],
        8: [1,1,1,1,1,1,1],
        9: [1,1,1,1,0,1,1],
    }

    # saída esperada em 4 bits (binário do dígito)
    targets_by_digit = {
        0: [0,0,0,0],
        1: [0,0,0,1],
        2: [0,0,1,0],
        3: [0,0,1,1],
        4: [0,1,0,0],
        5: [0,1,0,1],
        6: [0,1,1,0],
        7: [0,1,1,1],
        8: [1,0,0,0],
        9: [1,0,0,1],
    }

   
    X = []
    y = []
    for d in range(10):
        X.append(segments_by_digit[d])
        y.append(targets_by_digit[d])

    X = np.array(X, dtype=float)  
    y = np.array(y, dtype=float) 

    net = MLP(n_input=7, n_hidden=5, n_output=4, lr=0.5, seed=1)

    print("=== Treinando 7 segmentos ===")
    net.train(X, y, epochs=20000, print_every=4000)

    print("\n=== Teste sem ruído ===")
    preds = net.predict(X)
    for digit, (inp, p, target) in enumerate(zip(X, preds, y)):
        rounded = np.round(p)
        print(f"Digito {digit}")
        print(f"  Entrada        : {inp.tolist()}")
        print(f"  Previsto bruto : {np.round(p,3)}")
        print(f"  Previsto round : {rounded.astype(int).tolist()}")
        print(f"  Alvo           : {target.astype(int).tolist()}")
        print("-")

    print("\n=== Teste com ruído (flip_prob=0.2) ===")
    rng = np.random.default_rng(123)
    X_noisy = add_bitflip_noise(X, flip_prob=0.2, rng=rng)
    preds_noisy = net.predict(X_noisy)

    for digit, (clean, noisy, p) in enumerate(zip(X, X_noisy, preds_noisy)):
        rounded = np.round(p)
        print(f"Digito real {digit}")
        print(f"  Limpo   : {clean.tolist()}")
        print(f"  Ruidoso : {noisy.tolist()}")
        print(f"  Previsto bruto : {np.round(p,3)}")
        print(f"  Previsto round : {rounded.astype(int).tolist()}")
        print("-")
