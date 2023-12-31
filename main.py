import torch
import time
import argparse

from utils import *
from model import LSTMModel_serial
from torch.utils.data import DataLoader, Dataset

# 定义训练函数
def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("shit")
        # print(data.shape)

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    end = time.time()
    # print(f"epoch time: {end - start} seconds")
    return train_loss / len(train_loader)

# python main.py --end 2018-06-31 --device cuda
# python main.py --end 2018-12-32 --device cuda
# python main.py --end 2019-06-31 --device cuda
# python main.py --end 2019-12-32 --device cuda
# python main.py --end 2020-06-31 --device cuda
# python main.py --end 2020-12-32 --device cuda
# python main.py --end 2021-06-31 --device cuda
# python main.py --end 2021-12-32 --device cuda
# python main.py --end 2022-06-31 --device cuda
# python main.py --end 2022-12-32 --device cuda
# python main.py --end 2023-06-31 --device cuda
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=str, help="end date", default="2018-06-31")
    parser.add_argument("--device", type=str, help="choose cuda or cpu", default="cpu")
    args = parser.parse_args()


    file_list_init = get_file_list('./data/daypoint/')
    file_list = []
    test_end = args.end # '2018-06-31'
    device_name = args.device

    for i in range (len(file_list_init)):
        if file_list_init[i].split('/')[-1].split('.')[0] <= test_end:
            file_list.append(file_list_init[i])
    file_list = file_list[:-2] # prevent data leaking

    train_len = int(len(file_list) * 4 / 5)
    train_list = file_list[:train_len]
    test_list = file_list[train_len:]
    print(train_list)
    print(test_list)

    train_dataset = single_dataset(train_list)
    test_dataset = single_dataset(test_list)

    # 设置超参数
    input_size = 84          # six feature in one time step
    hidden_size = 30
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 1024
    num_layers = 1

    # 创建数据集和数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # 创建模型和优化器
    model = LSTMModel_serial(input_size, hidden_size, num_layers=num_layers, output_size=output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device(device_name)
    model.to(device)


    # 训练模型
    best_test_loss = 0
    best_train_loss = 0
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, train_loader, device)
        if best_train_loss > train_loss: best_train_loss = train_loss
        # print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        model.eval()
        test_loss = 0
        test_output = []
        test_target = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_output.append(output)
                test_target.append(target)

            pred = torch.concat(test_output).squeeze()
            true = torch.concat(test_target).squeeze()
            test_loss = loss_fn(pred, true)
        if(epoch % 10 == 0):
            print(pred)
            print(true)
        print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"./data/result/{test_end}_model.pt")
    
    print(f"best train loss {best_train_loss}, best test loss {best_test_loss}")

    with open("./data/result/loss.log", 'a') as f:
        f.write(f'\nDate {test_end}')
        f.write(f'\nBest Train Loss: {best_train_loss:.5f}')
        f.write(f'\nBest Test Loss: {best_test_loss:.5f}')