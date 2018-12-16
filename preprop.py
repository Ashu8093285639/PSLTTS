import pandas as pd

dataset = pd.read_csv('dataset_minyak_kelapa_sawit.csv')
dataset_value = dataset.iloc[:, 1].values

normalized_dataset = list()
for x in range(len(dataset_value)):
    norm_value = (dataset_value[x] - min(dataset_value)) / (
        max(dataset_value) - min(dataset_value))
    normalized_dataset.append(norm_value)

time_series_dataset = [
    normalized_dataset[:len(normalized_dataset) - 5],
    normalized_dataset[1:len(normalized_dataset) - 4],
    normalized_dataset[2:len(normalized_dataset) - 3],
    normalized_dataset[3:len(normalized_dataset) - 2],
    normalized_dataset[4:len(normalized_dataset) - 1],
    normalized_dataset[5:]]

# show_time_series_dataset
print("Time Series")
print(pd.DataFrame([list(i) for i in zip(*time_series_dataset)]))
