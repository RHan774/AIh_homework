import numpy as np

loadData_x = np.load('E:\桌面\学习\专业课\人工智能\lab\Week3-BP\dataset\sin_data\\train_data_x.npy')
output_x = "x.txt"
loadData_y = np.load('E:\桌面\学习\专业课\人工智能\lab\Week3-BP\dataset\sin_data\\train_data_y.npy')
output_y = "y.txt"
np.set_printoptions(threshold=np.inf)

with open(output_x, 'w') as f:
    print("----type----", file = f)
    print(type(loadData_x), file = f)
    print("----shape----", file = f)
    print(loadData_x.shape, file = f)
    print("----data----", file = f)
    print(loadData_x, file = f)
    
with open(output_y, 'w') as f:
    print("----type----", file = f)
    print(type(loadData_y), file = f)
    print("----shape----", file = f)
    print(loadData_y.shape, file = f)
    print("----data----", file = f)
    print(loadData_y, file = f)
# print(loadData[50])
# loadData.to_excel(r"output.xlsx", sheet_name="train_data", index= False,encoding="utf-8")