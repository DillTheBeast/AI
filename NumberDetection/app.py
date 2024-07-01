import torch
from model import SimpleNN, load_data, train_model, test_model

def main():
    # Load the data
    train_loader, test_loader = load_data()

    # Initialize the model
    model = SimpleNN()
    
    # Train the model
    train_model(model, train_loader, num_epochs=5)
    
    # Test the model
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
