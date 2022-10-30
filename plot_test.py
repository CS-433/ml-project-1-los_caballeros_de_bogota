from src.utils import plot_performance
import pickle
from src.helpers import load_csv_data, create_csv_submission
from src.data_processing import filter_data
import numpy as np 

DATA_PATH = "data/"

with open('src/best_models_10000.pkl', 'rb') as f:
    models = pickle.load(f)

prop = [0.104492, 0.030248, 0.011808, 0.005908, 0.29516, 0.279928, 0.189708, 0.082748]
global_acc = 0
global_f1 = 0

i = 0
for key1, value1 in models.items():
    for key2, value2 in value1.items():
        model = models[key1][key2]
        #plot_performance(model)
        
        global_acc += model.acc_te[-1]*prop[i]
        global_f1 += model.f1[-1]*prop[i]
        i += 1

print("The combined models achieves {:.4f} global accuracy and {:.4f} global F1 score on test set.\n".format(global_acc, global_f1))



# Create submission:
y, x, ids = load_csv_data(DATA_PATH + "test.csv", sub_sample=False)
data = filter_data(y, x, ids)
y_pred = np.array([])
ids_pred = np.array([])

for key1, value1 in data.items():
    for key2, value2 in value1.items():
        
        model = models[key1][key2]
        x = data[key1][key2]['x']
        
        y_pred = np.concatenate((y_pred, model.predict(x, eval_mode=True)))
        ids_pred = np.concatenate((ids_pred, data[key1][key2]['ids']))

y_pred[y_pred == 0] = -1
create_csv_submission(ids_pred, y_pred, DATA_PATH + "submission.csv")
print("Submission saved at 'data/submission.csv'.")