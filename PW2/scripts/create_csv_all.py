from pathlib import Path
import re
import pandas as pd
import json

output_dir = Path('./csvs/')
output_dir.mkdir(exist_ok=True, parents=True)
dataset_headers = {
    "car": ["buying","maint","doors","persons","lug_boot","safety"],
    "kr-vs-kp": ["bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd","hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr","skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg"],
    "iris": ["sepal_length","sepal_width","petal_length","petal_width"]
}

for dataset in Path('models/').glob('*/'):
    dataset_name = dataset.parts[-1]

    dfs = {
        "RF": [['F,NT,fit_time,predict_time,fit_accuracy,predict_accuracy'] + dataset_headers.get(dataset_name)],
        "DF": [['F,NT,fit_time,predict_time,fit_accuracy,predict_accuracy'] + dataset_headers.get(dataset_name)],
    }

    train_csv_df = pd.read_csv(dataset / 'DF_train.csv')
    train_csv_rf = pd.read_csv(dataset / 'RF_train.csv')
    # train_csv = pd.concat((train_csv_df, train_csv_rf))
    test_csv = pd.read_csv(dataset / 'sorted_test.csv')
    # del train_csv_df, train_csv_rf

    for df in [train_csv_df, train_csv_rf]:
        for i, row in df.iterrows():
            name, F, NT, fit_time, train_acc = list(row)
            folder_name = f'{name}_F-{F}_NT-{NT}'
            counts_path = dataset / folder_name / 'feat_counts.json'
            with open(counts_path, 'r') as f:
                counts = json.load(f)
            test_data = test_csv[test_csv['name'] == folder_name]
            if len(test_data) > 0:
                test_data = test_data.iloc[0]
            else:
                continue
            line = [F, NT, round(fit_time, 3), round(test_data['predict_time(s)'], 3), round(train_acc, 3), round(test_data['test_acc'], 3)]
            counts = [i[1] for i in sorted(counts.items(), key=lambda x: dataset_headers.get(dataset_name).index(x[0]))]
            dfs[name].append(list(map(str, line + counts)))

    for name in dfs:
        with open(output_dir / f'{dataset_name}_{name}.csv', 'w') as f:
            f.write('\n'.join(map(lambda x: ','.join(x), dfs[name])))
