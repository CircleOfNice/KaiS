import os
import pandas as pd
print(os.getcwd())

file_names = ['taskdata_1.json','taskdata_2.json', 'taskdata_3.json', 'taskdata_4.json', 'taskdata_5.json', 'taskdata_6.json'] 
for file_name in file_names:
    print('File ')
    path = os.path.join(os.getcwd(), 'data', file_name, 'task.csv')
    df = pd.read_csv(path)
    print(df.head())
    print('df[taskName].value_counts()) : ', df['taskName'].value_counts())
    print('df[scheduled_node].value_counts() : ', df['scheduled_node'].value_counts())
    print('df[taskName].nunique() : ', df['taskName'])
    import matplotlib.pylab as plt
    from matplotlib.pyplot import figure
    path = os.path.join(os.getcwd(), 'data', file_name,)
    #figure(figsize=(8, 6), dpi=300)
    for col in df.columns:
        try:
            #print(col)
            df[col].plot(figsize=(20, 10),title=col)
            
            
            plt.savefig(os.path.join(path, col+'.jpg'))
            #plt.show()
            plt.close()
        except:
            continue
            plt.close()