import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from .layer import Layer
from .loss import Loss
import os
from tqdm import tqdm

class FFNN:
    def __init__(self, loss='mse',l1_lambda=0.0, l2_lambda=0.0):
        self.loss_name = loss # buat nyimpen nama loss function
        self.layers = []
        self.loss_function = Loss.get_loss(loss)
        self.layer_sizes = []
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda 
    
    def add(self, input_size, output_size, activation='linear', 
            weight_initializer='uniform', **initializer_params):
        
        # Cek kalo misal ini layer pertama atau bukan
        if not self.layers:
            self.layer_sizes.append(input_size)
        
        self.layer_sizes.append(output_size)
        
        # Ngebuat layer baru ke neural network sama nambahin ke list layers
        layer = Layer(input_size, output_size, activation, 
                  weight_initializer, **initializer_params)

        self.layers.append(layer)
        return self
    
    def forward(self, X):
        output = X

        # Iterasi untuk setiap layer
        for layer in self.layers:
            # Ngelakuin forward propagation dengan inputnya berupa hasil output layer sebelumnya
            output = layer.forward(output)

        # Ngembaliin output akhir dari neural network
        return output
    
    def backward(self, y_true, y_pred):
        # Cek kalo loss functionnya sama dengan softmax dan activation functionnya sama dengan categorical crossentropy di layer terakhir
        # Karena kalo iya ada special case buat ngitung gradient awalnya
        if (self.layers[-1].activation.__class__.__name__ == 'Softmax' and 
            self.loss_function.__name__ == 'CategoricalCrossEntropy'):
            dvalues = self.loss_function.softmax_derivative(y_true, y_pred)
        else:
            dvalues = self.loss_function.derivative(y_true, y_pred)
        
        # Meneruskan gradien dari belakang ke depan (reversed)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
        
        return dvalues
    
    def update_weights(self, learning_rate):
        for layer in self.layers:
            # Menambahkan regularisasi L1 dan L2 ke gradien bobot
            if self.l1_lambda > 0:
                l1_grad = self.l1_lambda * np.sign(layer.W)
                layer.dW += l1_grad
            
            if self.l2_lambda > 0:
                l2_grad = 2 * self.l2_lambda * layer.W
                layer.dW += l2_grad
                
            # Update weights with gradients
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
    
    def fit(self, X_train, y_train, batch_size=32, learning_rate=0.01, 
            epochs=100, validation_data=None, verbose=1):
        
        # Dictionary buat nyimpen history training
        history = {
            'loss': [],
            'val_loss': [] if validation_data else None
        }
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle data setiap epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                # Ambil batch data
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Lakuin forward propagation
                y_pred = self.forward(X_batch)
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

                # Hitung lossnya
                batch_loss = self.loss_function.calculate(y_batch, y_pred)

                # Tambahkan regularisasi L1 dan L2 untuk Batch Loss
                if self.l1_lambda > 0 or self.l2_lambda > 0:
                    l1_penalty = self.l1_lambda * sum(np.sum(np.abs(layer.W)) for layer in self.layers)
                    l2_penalty = self.l2_lambda * sum(np.sum(layer.W**2) for layer in self.layers)
                    batch_loss += l1_penalty + l2_penalty

                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples

                # Lakuin backward propagation
                self.backward(y_batch, y_pred)
                
                # Update bobot
                self.update_weights(learning_rate)
            
            # Simpan loss ke history
            history['loss'].append(epoch_loss)
            
            # Hitung validation loss kalo ada
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function.calculate(y_val, y_val_pred)

                # Tambahkan regularisasi L1 dan L2 untuk Validation Loss
                if self.l1_lambda > 0 or self.l2_lambda > 0:
                    l1_penalty_val = self.l1_lambda * sum(np.sum(np.abs(layer.W)) for layer in self.layers)
                    l2_penalty_val = self.l2_lambda * sum(np.sum(layer.W**2) for layer in self.layers)
                    val_loss += l1_penalty_val + l2_penalty_val

                history['val_loss'].append(val_loss)
            
            # Tunjukin progress kalo ada
            if verbose == 1:
                val_msg = f", val_loss: {val_loss:.4f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}{val_msg}")
                
                # Progress bar
                progress = int(30 * (epoch + 1) / epochs)
                bar = "[" + "=" * progress + ">" + " " * (30 - progress - 1) + "]"
                print(bar, end="\r")
        
        if verbose == 1:
            print()
            
        return history
    
    def visualize_network(self, figsize=(10, 6)):
        """Memvisualisasikan struktur jaringan saraf dengan bobot-bobotnya"""
        fig, ax = plt.subplots(figsize=figsize)
        G = nx.DiGraph()
        
        # Tambahkan node untuk setiap layer
        pos = {}
        node_labels = {}
        node_colors = []
        
        # Jarak antar layer yang merata
        layer_spacing = 1.0
        
        # Tambahkan node-node layer input
        input_layer_size = self.layer_sizes[0]
        for i in range(input_layer_size):
            node_id = f"input_{i}"
            G.add_node(node_id)
            pos[node_id] = (0, (input_layer_size-1)/2 - i)
            node_labels[node_id] = f"I{i}"
            node_colors.append('skyblue')
        
        # Tambahkan node-node layer tersembunyi dan output
        for l, layer_size in enumerate(self.layer_sizes[1:], 1):
            for i in range(layer_size):
                node_id = f"layer{l}_{i}"
                G.add_node(node_id)
                pos[node_id] = (l * layer_spacing, (layer_size-1)/2 - i)
                
                # Layer terakhir adalah layer output
                if l == len(self.layer_sizes) - 1:
                    node_labels[node_id] = f"O{i}"
                    node_colors.append('lightgreen')
                else:
                    node_labels[node_id] = f"H{l}_{i}"
                    node_colors.append('lightsalmon')
        
        # Tambahkan edge dengan bobot
        edge_labels = {}  # Dictionary untuk menyimpan label edge
        
        print("Add edges with weights: ")
        for l, layer in enumerate(self.layers):
            source_size = self.layer_sizes[l]
            target_size = self.layer_sizes[l+1]
            
            # Tambahkan edge antara layer ini dan layer berikutnya
            for i in tqdm(range(source_size), desc=f"Layer {l} to Layer {l+1}", leave=False):
                for j in range(target_size):
                    source = f"input_{i}" if l == 0 else f"layer{l}_{i}"
                    target = f"layer{l+1}_{j}"
                    
                    weight = layer.W[i, j]
                    gradient = layer.dW[i, j] if hasattr(layer, 'dW') and layer.dW is not None else 0
                    
                    # Ketebalan edge berdasarkan besarnya bobot
                    width = min(abs(weight) * 3, 3)
                    
                    G.add_edge(source, target, weight=weight, gradient=gradient, width=width)
                    
                    # Tambahkan bobot terformat sebagai label edge
                    edge_labels[(source, target)] = f"W:{weight:.2f}\nG:{gradient:.2f}"
        
        print("Draw network")
        # Gambar jaringan
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        gradients = [G[u][v]['gradient'] for u, v in edges]
        widths = [G[u][v]['width'] for u, v in edges]
        
        print("Normalize weights to map to colors")
        # Normalisasi bobot untuk pemetaan ke warna
        if weights:
            max_weight = max(abs(w) for w in weights) if weights else 1
            edge_colors = [plt.cm.RdBu(0.5 * (1 + w/max_weight)) for w in weights]
        else:
            max_weight = 1
            edge_colors = ['gray']

        print("Draw edges")    
        # Gambar edge
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=widths, 
            edge_color=edge_colors, arrows=True, arrowsize=10, ax=ax
        )
        
        print("Draw edge labels")
        # Gambar label edge (bobot)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, 
            font_size=8,
            font_color='black',
            font_family='sans-serif',
            ax=ax
        )
        print("Draw nodes")
        # Gambar node
        nx.draw_networkx_nodes(
            G, pos, node_size=500, 
            node_color=node_colors, edgecolors='black', ax=ax
        )
        
        print("Draw node labels")
        # Gambar label
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
        
        print("Draw legend")
        # Tambahkan legenda untuk warna edge
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(-max_weight, max_weight), 
            cmap=plt.cm.RdBu
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Weight Value')
        
        ax.set_title("Neural Network Structure with Weights")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_weight_distribution(self, layers=None):
        """Memvisualisasikan distribusi bobot pada jaringan saraf"""
        if layers is None:
            layers = list(range(len(self.layers)))
            
        n_layers = len(layers)
        fig, axs = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        # Menangani kasus layer tunggal
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist.")
                continue
                
            layer = self.layers[layer_idx]
            ax = axs[i]
            
            # Plot distribusi bobot
            weights = layer.W.flatten()
            ax.hist(weights, bins=50, alpha=0.7)
            ax.set_title(f"Layer {layer_idx+1} Weight Distribution")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            
            # Tambahkan garis vertikal untuk mean dan std
            mean = np.mean(weights)
            std = np.std(weights)
            ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean = {mean:.4f}')
            ax.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean + Std = {mean+std:.4f}')
            ax.axvline(mean - std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean - Std = {mean-std:.4f}')
            
            # Tambahkan distribusi bias sebagai inset
            biases = layer.b.flatten()
            if len(biases) > 1: # Hanya tambahkan inset jika ada banyak bias
                inset_ax = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
                inset_ax.hist(biases, bins=min(20, len(biases)), alpha=0.7, color='orange')
                inset_ax.set_title("Bias Distribution")
                inset_ax.tick_params(axis='both', which='both', labelsize=6)
            
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def visualize_gradient_distribution(self, layers=None):
        """Memvisualisasikan distribusi gradien pada jaringan saraf"""
        if layers is None:
            layers = list(range(len(self.layers)))
            
        n_layers = len(layers)
        fig, axs = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        # Menangani kasus layer tunggal
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist.")
                continue
                
            layer = self.layers[layer_idx]
            ax = axs[i]
            
            # Periksa apakah gradien ada
            if not hasattr(layer, 'dW') or layer.dW is None:
                ax.text(0.5, 0.5, "No gradient data (need backward pass)", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot distribusi gradien
            gradients = layer.dW.flatten()
            ax.hist(gradients, bins=50, alpha=0.7, color='purple')
            ax.set_title(f"Layer {layer_idx+1} Weight Gradient Distribution")
            ax.set_xlabel("Gradient Value")
            ax.set_ylabel("Frequency")
            
            # Tambahkan garis vertikal untuk mean dan std
            mean = np.mean(gradients)
            std = np.std(gradients)
            ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean = {mean:.4f}')
            ax.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean + Std = {mean+std:.4f}')
            ax.axvline(mean - std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean - Std = {mean-std:.4f}')
            
            # Tambahkan distribusi gradien bias sebagai inset
            if hasattr(layer, 'db') and layer.db is not None:
                bias_gradients = layer.db.flatten()
                if len(bias_gradients) > 1:  # Only add inset if there are multiple bias gradients
                    inset_ax = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
                    inset_ax.hist(bias_gradients, bins=min(20, len(bias_gradients)), 
                                 alpha=0.7, color='orange')
                    inset_ax.set_title("Bias Gradient Distribution")
                    inset_ax.tick_params(axis='both', which='both', labelsize=6)
            
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath):
        # Pastiin kalo direktori buat nyimpen ada
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Cek jenis loss function yang digunakan
        if hasattr(self, 'loss_name'):
            # Jika loss_name sudah disimpan saat inisialisasi 
            loss_id = self.loss_name
        else:
            print(f"Loss function type: {type(self.loss_function)}")
            
            # Default ke categorical_crossentropy jika tidak bisa ditentukan
            loss_id = 'categorical_crossentropy'
            
            # Coba ambil nama dari instance
            try:
                loss_name = self.loss_function.__class__.__name__
                # Pemetaan nama ke identifier
                loss_mapping = {
                    'MeanSquaredError': 'mse',
                    'CategoricalCrossEntropy': 'categorical_crossentropy',
                    'BinaryCrossEntropy': 'binary_crossentropy'
                }
                if loss_name in loss_mapping:
                    loss_id = loss_mapping[loss_name]
            except:
                # Jika gagal, gunakan default
                pass
                
        # Simpan model ke file dengan identifier loss function
        model_data = {
            'layer_sizes': self.layer_sizes,
            'layers': self.layers,
            'loss_function': loss_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
                
        print(f"Model saved to {filepath} with loss: {loss_id}")
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load model dari file
        model = cls(loss=model_data['loss_function'])
        model.layer_sizes = model_data['layer_sizes']
        model.layers = model_data['layers']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self):
        """Mencetak ringkasan arsitektur jaringan saraf"""
        print("Neural Network Architecture Summary")
        print("==================================")
        print(f"Total layers: {len(self.layers)}")
        
        # Menghitung total parameter yang dapat dilatih
        total_params = 0
        
        print("\nLayer Details:")
        print("--------------")
        # Menampilkan ukuran input
        print(f"Input shape: ({self.layer_sizes[0]})")
        
        # Iterasi melalui setiap layer untuk menampilkan detailnya
        for i, layer in enumerate(self.layers):
            layer_name = f"Layer {i+1}"
            activation = layer.activation.__class__.__name__
            # Menghitung jumlah parameter (bobot + bias)
            params = layer.W.size + layer.b.size
            total_params += params
            
            # Mencetak informasi layer: ukuran input→output, fungsi aktivasi, dan jumlah parameter
            print(f"  {layer_name}: {self.layer_sizes[i]} → {self.layer_sizes[i+1]} | " +
                f"Activation: {activation} | Parameters: {params}")
        
        # Menampilkan total parameter yang dapat dilatih
        print("\nTotal trainable parameters:", total_params)