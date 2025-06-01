import pickle
import numpy as np

def detecting_fake_news(var):
    try:
        # Load the newly trained model
        load_model = pickle.load(open('final_model_new.sav', 'rb'))
        prediction = load_model.predict([var])
        prob = load_model.predict_proba([var])
        
        if prediction[0] == 'FAKE':
            print(f"The news headline is FAKE with probability: {prob[0][0]:.2f}")
        else:
            print(f"The news headline is REAL with probability: {prob[0][1]:.2f}")
            
    except FileNotFoundError:
        print("Model file not found. Please run classifier_simple.py first to train the model.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    print("Fake News Detection System")
    print("Enter 'quit' to exit")
    
    while True:
        var = input("Please enter the news text you want to verify: ")
        if var.lower() == 'quit':
            break
        print(f"You entered: {var}")
        detecting_fake_news(var)
        print("-" * 50)
