import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=1.0, seed=0):
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

    def train(self, X, y, epochs=200_000, print_every=20_000):
        for ep in range(1, epochs + 1):
            a1, z1, y_pred, z2 = self.forward(X)
            loss = mse(y, y_pred)
            self.backward(X, y, a1, z1, y_pred, z2)

            if ep % print_every == 0:
                print(f"[epoch {ep}] loss = {loss:.10f}")

    def predict(self, X):
        _, _, y_pred, _ = self.forward(X)
        return y_pred

if __name__ == "__main__":
    # tabela verdade do XOR
    X_xor = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ], dtype=float)

    y_xor = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    xor_net = MLP(n_input=2, n_hidden=2, n_output=1, lr=1.0, seed=0)

    print("=== Treinando XOR ===")
    xor_net.train(X_xor, y_xor, epochs=200_000, print_every=20_000)

    print("\n=== Teste XOR ===")
    preds = xor_net.predict(X_xor)

    for x, p, y_true in zip(X_xor, preds, y_xor):
        arredondado = int(np.round(p[0]))
        print(f"Entrada {x} -> saida_pred {p[0]:.6f} -> saida_round {arredondado} -> alvo {int(y_true[0])}")
