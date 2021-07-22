import pandas as pd
data = pd.HDFStore('output.hdf5')
_data =data['table'].iloc[0]
trees = _data['trees']
def print_dict(d):
    for k,v in d.items():
        print(f'----{k}----')
        print(v)
def iterative_print(tr):
    if 'children' in tr.keys():
        for i in tr['children']:
            print_dict(i)
            iterative_print(i)

print_dict(trees[1])
iterative_print(trees[1])

#f = open('txt_out.txt','w')
#for i in range(10):
#    contents = list(data['table'].iloc[i])[:-1]
#    for c in contents:
#        f.write(f'{c}\n')
#    f.write('\n')
#f.close()
