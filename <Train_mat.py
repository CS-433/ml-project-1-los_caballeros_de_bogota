from src.helpers import load_csv_data

data_path_te = "data/test.csv"
data_path_tr = "data/train.csv"
data_path_te = "data/sample-submissive.csv"

yb_te, input_data_te, ids_te =load_csv_data(data_path_te, sub_sample=False)

yb_tr, input_data_tr, ids_tr =load_csv_data(data_path_tr, sub_sample=False)

