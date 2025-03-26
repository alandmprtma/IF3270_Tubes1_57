# model.py
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
    """Feed-Forward Neural Network implementation with all required functionalities"""
    def __init__(self, loss='mse'):
        self.layers = []
        self.loss_function = Loss.get_loss(loss)
        self.layer_sizes = []
    
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
        #Memperbarui bobot untuk semua layer menggunakan gradien yang dihitung
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
    
    def fit(self, X_train, y_train, batch_size=32, learning_rate=0.01, 
            epochs=100, validation_data=None, verbose=1):
        history = {
            'loss': [],
            'val_loss': [] if validation_data else None
        }
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle dataset for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                # Get batch data
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

                # Calculate loss
                batch_loss = self.loss_function.calculate(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples

                # Backward pass
                self.backward(y_batch, y_pred)
                
                # Update weights
                self.update_weights(learning_rate)
            
            # Store training loss
            history['loss'].append(epoch_loss)
            
            # Calculate validation loss if provided
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function.calculate(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
            
            # Display progress
            if verbose == 1:
                val_msg = f", val_loss: {val_loss:.4f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}{val_msg}")
                
                # Simple progress bar
                progress = int(30 * (epoch + 1) / epochs)
                bar = "[" + "=" * progress + ">" + " " * (30 - progress - 1) + "]"
                print(bar, end="\r")
        
        # Final newline after progress bar
        if verbose == 1:
            print()
            
        return history
    
    def visualize_network(self, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        G = nx.DiGraph()
        
        # Add nodes for each layer
        pos = {}
        node_labels = {}
        node_colors = []
        
        # Space out layers evenly
        layer_spacing = 1.0
        
        # Add input layer nodes
        input_layer_size = self.layer_sizes[0]
        for i in range(input_layer_size):
            node_id = f"input_{i}"
            G.add_node(node_id)
            pos[node_id] = (0, (input_layer_size-1)/2 - i)
            node_labels[node_id] = f"I{i}"
            node_colors.append('skyblue')
        
        # Add hidden and output layer nodes
        for l, layer_size in enumerate(self.layer_sizes[1:], 1):
            for i in range(layer_size):
                node_id = f"layer{l}_{i}"
                G.add_node(node_id)
                pos[node_id] = (l * layer_spacing, (layer_size-1)/2 - i)
                
                # Last layer is output layer
                if l == len(self.layer_sizes) - 1:
                    node_labels[node_id] = f"O{i}"
                    node_colors.append('lightgreen')
                else:
                    node_labels[node_id] = f"H{l}_{i}"
                    node_colors.append('lightsalmon')
        
        # Add edges with weights
        edge_labels = {}  # Dictionary to store edge labels
        
        print("Adding edges with weights:")
        for l, layer in enumerate(self.layers):
            source_size = self.layer_sizes[l]
            target_size = self.layer_sizes[l+1]
            
            # Add edges between this layer and next
            for i in tqdm(range(source_size), desc=f"Layer {l} to Layer {l+1}", leave=False):
                for j in range(target_size):
                    source = f"input_{i}" if l == 0 else f"layer{l}_{i}"
                    target = f"layer{l+1}_{j}"
                    
                    weight = layer.W[i, j]
                    gradient = layer.dW[i, j] if hasattr(layer, 'dW') and layer.dW is not None else 0
                    
                    # Edge width based on weight magnitude
                    width = min(abs(weight) * 3, 3)
                    
                    G.add_edge(source, target, weight=weight, gradient=gradient, width=width)
                    
                    # Add formatted weight as edge label
                    edge_labels[(source, target)] = f"{weight:.2f}"
        
        print("Draw network")
        # Draw the network
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        gradients = [G[u][v]['gradient'] for u, v in edges]
        widths = [G[u][v]['width'] for u, v in edges]
        
        print("Normalize weights to map to colors")
        # Normalize weights to map to colors
        if weights:
            max_weight = max(abs(w) for w in weights) if weights else 1
            edge_colors = [plt.cm.RdBu(0.5 * (1 + w/max_weight)) for w in weights]
        else:
            max_weight = 1
            edge_colors = ['gray']

        print("Draw edges")    
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=widths, 
            edge_color=edge_colors, arrows=True, arrowsize=10, ax=ax
        )
        
        print("Draw edge labels")
        # Draw edge labels (weights)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, 
            font_size=8,
            font_color='black',
            font_family='sans-serif',
            ax=ax
        )
        print("Draw nodes")
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_size=500, 
            node_color=node_colors, edgecolors='black', ax=ax
        )
        
        print("Draw labels")
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
        
        print("Draw legend")
        # Add legend for edge colors
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
        if layers is None:
            layers = list(range(len(self.layers)))
            
        n_layers = len(layers)
        fig, axs = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        # Handle single layer case
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist.")
                continue
                
            layer = self.layers[layer_idx]
            ax = axs[i]
            
            # Plot weight distribution
            weights = layer.W.flatten()
            ax.hist(weights, bins=50, alpha=0.7)
            ax.set_title(f"Layer {layer_idx+1} Weight Distribution")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            
            # Add mean and std as vertical lines
            mean = np.mean(weights)
            std = np.std(weights)
            ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean = {mean:.4f}')
            ax.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean + Std = {mean+std:.4f}')
            ax.axvline(mean - std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean - Std = {mean-std:.4f}')
            
            # Add bias distribution as inset
            biases = layer.b.flatten()
            if len(biases) > 1:  # Only add inset if there are multiple biases
                inset_ax = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
                inset_ax.hist(biases, bins=min(20, len(biases)), alpha=0.7, color='orange')
                inset_ax.set_title("Bias Distribution")
                inset_ax.tick_params(axis='both', which='both', labelsize=6)
            
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_gradient_distribution(self, layers=None):
        if layers is None:
            layers = list(range(len(self.layers)))
            
        n_layers = len(layers)
        fig, axs = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        # Handle single layer case
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist.")
                continue
                
            layer = self.layers[layer_idx]
            ax = axs[i]
            
            # Check if gradients exist
            if not hasattr(layer, 'dW') or layer.dW is None:
                ax.text(0.5, 0.5, "No gradient data (need backward pass)", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot gradient distribution
            gradients = layer.dW.flatten()
            ax.hist(gradients, bins=50, alpha=0.7, color='purple')
            ax.set_title(f"Layer {layer_idx+1} Weight Gradient Distribution")
            ax.set_xlabel("Gradient Value")
            ax.set_ylabel("Frequency")
            
            # Add mean and std as vertical lines
            mean = np.mean(gradients)
            std = np.std(gradients)
            ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean = {mean:.4f}')
            ax.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean + Std = {mean+std:.4f}')
            ax.axvline(mean - std, color='g', linestyle='dashed', linewidth=1, 
                      label=f'Mean - Std = {mean-std:.4f}')
            
            # Add bias gradient distribution as inset
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
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Extract model state
        model_data = {
            'layer_sizes': self.layer_sizes,
            'layers': self.layers,
            'loss_function': self.loss_function.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model
        model = cls(loss=model_data['loss_function'])
        model.layer_sizes = model_data['layer_sizes']
        model.layers = model_data['layers']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self):
        """Print a summary of the neural network architecture"""
        print("Neural Network Architecture Summary")
        print("==================================")
        print(f"Total layers: {len(self.layers)}")
        
        total_params = 0
        
        print("\nLayer Details:")
        print("--------------")
        print(f"Input shape: ({self.layer_sizes[0]})")
        
        for i, layer in enumerate(self.layers):
            layer_name = f"Layer {i+1}"
            activation = layer.activation.__class__.__name__
            params = layer.W.size + layer.b.size
            total_params += params
            
            print(f"  {layer_name}: {self.layer_sizes[i]} → {self.layer_sizes[i+1]} | " +
                  f"Activation: {activation} | Parameters: {params}")
        
        print("\nTotal trainable parameters:", total_params)