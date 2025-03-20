import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ffnn.model import FFNN

def main():
    print("FFNN Neural Network Visualization Demo")
    print("======================================")
    
    use_iris = True
    
    if use_iris:
        print("\nLoading iris dataset...")
        iris = load_iris()
        X = iris.data
        y = iris.target.reshape(-1, 1)
        feature_count = X.shape[1] 
        class_count = 3 
    else:
        print("\nLoading digits dataset...")
        digits = load_digits()
        X = digits.data
        y = digits.target.reshape(-1, 1)
        feature_count = X.shape[1]  
        class_count = 10  
        
        # Option: Use PCA to reduce feature dimensions for visualization
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=8)
        # X = pca.fit_transform(X)
        # feature_count = 8
    
    # Scale features and one-hot encode targets
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset: {X.shape[0]} samples, {feature_count} features, {class_count} classes")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create a smaller neural network
    print("\nCreating neural network...")
    input_size = feature_count
    hidden_size = 5  
    output_size = class_count
    
    model = FFNN(loss='categorical_crossentropy')
    model.add(input_size=input_size, output_size=hidden_size, activation='relu',
              weight_initializer='normal', mean=0, std=0.1)
    model.add(input_size=hidden_size, output_size=output_size, activation='softmax',
              weight_initializer='normal', mean=0, std=0.1)
    
    # Print model summary
    model.summary()
    
    # Visualize initial network structure
    print("\nVisualizing initial network structure...")
    model.visualize_network(figsize=(12, 8))
    
    # Visualize initial weight distribution
    print("\nVisualizing initial weight distribution...")
    model.visualize_weight_distribution()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        learning_rate=0.01,
        epochs=20,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Evaluate model on test set
    y_pred = model.forward(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred_classes == y_test_classes)
    
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [accuracy], color='green')
    plt.title(f'Test Accuracy: {accuracy:.4f}')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    # Visualize trained network structure
    print("\nVisualizing trained network structure...")
    model.visualize_network(figsize=(12, 8))
    
    # Visualize final weight distribution
    print("\nVisualizing final weight distribution...")
    model.visualize_weight_distribution()
    
    # Visualize gradient distribution
    print("\nVisualizing gradient distribution...")
    # First do a forward/backward pass to compute gradients
    y_pred = model.forward(X_test[:32])  # Use a batch
    model.backward(y_test[:32], y_pred)
    model.visualize_gradient_distribution()
    
    if not use_iris:
        # Visualize a few example predictions (only for digits)
        print("\nVisualizing example predictions...")
        plt.figure(figsize=(15, 5))
        for i in range(5):
            # Get a random sample
            idx = np.random.randint(0, len(X_test))
            x = X_test[idx]
            true_label = np.argmax(y_test[idx])
            
            # Make prediction
            pred = model.forward(x.reshape(1, -1))
            pred_label = np.argmax(pred)
            
            # Plot the digit
            plt.subplot(1, 5, i+1)
            plt.imshow(np.reshape(scaler.inverse_transform([x])[0], (8, 8)), 
                      cmap='gray')
            plt.title(f"True: {true_label}\nPred: {pred_label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\nDemo completed!")
    
if __name__ == "__main__":
    main()