import pickle

def load(model_path):
    model = pickle.load(open('model.pkl','rb'))
    return model


