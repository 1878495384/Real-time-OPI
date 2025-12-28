# 训练模型33333，通道数16,batch=4 动态, 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import pandas as pd
from model import StarNet  # 确保导入的是支持512x512的StarNet版本


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Custom dataset
class PhaseObjectDataset(Dataset):
    def __init__(self, image_dir, diffraction_dir):
        self.image_dir = image_dir
        self.diffraction_dir = diffraction_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.diffraction_files = [f for f in os.listdir(diffraction_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        # Ensure the number of original and diffraction images match
        assert len(self.image_files) == len(self.diffraction_files), "Mismatched number of images"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load diffraction image
        diffraction_path = os.path.join(self.diffraction_dir, self.diffraction_files[idx])
        diffraction_image = np.array(Image.open(diffraction_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
        diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0)

        # Load corresponding original image (phase object)
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.array(Image.open(image_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return diffraction_tensor, image_tensor  # Return both tensors

# Main program
image_dir = 'frame/ori4'  # Set your image directory
diffraction_dir = 'frame/dif4'  # Set your diffraction directory
dataset = PhaseObjectDataset(image_dir, diffraction_dir)

# Split the dataset
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the StarNet model
model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 使用支持512x512的StarNet版本
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

# Record losses
train_losses = []
test_losses = []

# Training function
def train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
    best_model_path = 'best_model_starnet16(d2-1).pth'  # 最佳模型权重保存路径

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0

        # Training phase
        for data, targets in train_dataloader:
            data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)

        # Testing phase
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, targets in test_dataloader:
                data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
                output = model(data)
                loss = criterion(output, targets)
                total_test_loss += loss.item()

        average_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(average_test_loss)

        # 保存最佳模型权重
        if average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch + 1} with Test Loss: {average_test_loss:.12f}")

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {average_train_loss:.12f}, '
              f'Test Loss: {average_test_loss:.12f}, '
              f'Time: {elapsed_time / 60:.2f} minutes')

    print(f"Training completed. Best model saved with Test Loss: {best_test_loss:.12f}")

# Train the network
train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)

# Save model weights
torch.save(model.state_dict(), 'best_model_starnet16(d2-1).pth')

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='orange', linestyle='-')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='blue', linestyle='-')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 设置 x 轴和 y 轴的范围，让它们从 0 开始
plt.xlim(0, num_epochs + 1)
plt.ylim(0, max(max(train_losses), max(test_losses)))
plt.show()

# Save loss values to Excel
loss_data = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Loss': train_losses, 'Test Loss': test_losses})
loss_data.to_excel('loss_values16(d2-1).xlsx', index=False)  # 保存到Excel文件




    

# # 训练模型331255，通道数16,batch=4 动态
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import time
# import pandas as pd
# from model import StarNet  # 确保导入的是支持512x512的StarNet版本


# # Set random seed for reproducibility
# torch.manual_seed(0)
# np.random.seed(0)

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# # Custom dataset
# class PhaseObjectDataset(Dataset):
#     def __init__(self, image_dir, diffraction_dir):
#         self.image_dir = image_dir
#         self.diffraction_dir = diffraction_dir
#         self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
#         self.diffraction_files = [f for f in os.listdir(diffraction_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

#         # Ensure the number of original and diffraction images match
#         assert len(self.image_files) == len(self.diffraction_files), "Mismatched number of images"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # Load diffraction image
#         diffraction_path = os.path.join(self.diffraction_dir, self.diffraction_files[idx])
#         diffraction_image = np.array(Image.open(diffraction_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
#         diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0)

#         # Load corresponding original image (phase object)
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         image = np.array(Image.open(image_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
#         image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

#         return diffraction_tensor, image_tensor  # Return both tensors

# # Main program
# image_dir = 'frame/ori4'  # Set your image directory
# diffraction_dir = 'frame/dif4'  # Set your diffraction directory
# dataset = PhaseObjectDataset(image_dir, diffraction_dir)

# # Split the dataset
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# # Create data loaders
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # Initialize the StarNet model
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 使用支持512x512的StarNet版本
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 100

# # Record losses
# train_losses = []
# test_losses = []

# # Training function
# def train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
#     best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
#     best_model_path = 'best_model_starnet(d2-1).pth'  # 最佳模型权重保存路径

#     for epoch in range(num_epochs):
#         start_time = time.time()
#         model.train()
#         total_train_loss = 0.0

#         # Training phase
#         for data, targets in train_dataloader:
#             data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, targets)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)

#         # Testing phase
#         model.eval()
#         total_test_loss = 0.0
#         with torch.no_grad():
#             for data, targets in test_dataloader:
#                 data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
#                 output = model(data)
#                 loss = criterion(output, targets)
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)

#         # 保存最佳模型权重
#         if average_test_loss < best_test_loss:
#             best_test_loss = average_test_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"New best model saved at epoch {epoch + 1} with Test Loss: {average_test_loss:.12f}")

#         elapsed_time = time.time() - start_time
#         print(f'Epoch [{epoch + 1}/{num_epochs}], '
#               f'Train Loss: {average_train_loss:.12f}, '
#               f'Test Loss: {average_test_loss:.12f}, '
#               f'Time: {elapsed_time / 60:.2f} minutes')

#     print(f"Training completed. Best model saved with Test Loss: {best_test_loss:.12f}")

# # Train the network
# train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)

# # Save model weights
# torch.save(model.state_dict(), 'best_model_starnet(d2-1).pth')

# # Plot loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='orange', linestyle='-')
# plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='blue', linestyle='-')
# plt.title('Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()

# # 设置 x 轴和 y 轴的范围，让它们从 0 开始
# plt.xlim(0, num_epochs + 1)
# plt.ylim(0, max(max(train_losses), max(test_losses)))
# plt.show()

# # Save loss values to Excel
# loss_data = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Loss': train_losses, 'Test Loss': test_losses})
# loss_data.to_excel('loss_values(d2-1).xlsx', index=False)  # 保存到Excel文件





# # 训练模型331255，通道数16,batch=4 静态
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import time
# import pandas as pd
# from model import StarNet  # 确保导入的是支持512x512的StarNet版本


# # Set random seed for reproducibility
# torch.manual_seed(0)
# np.random.seed(0)

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# # Custom dataset
# class PhaseObjectDataset(Dataset):
#     def __init__(self, image_dir, diffraction_dir):
#         self.image_dir = image_dir
#         self.diffraction_dir = diffraction_dir
#         self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
#         self.diffraction_files = [f for f in os.listdir(diffraction_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

#         # Ensure the number of original and diffraction images match
#         assert len(self.image_files) == len(self.diffraction_files), "Mismatched number of images"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # Load diffraction image
#         diffraction_path = os.path.join(self.diffraction_dir, self.diffraction_files[idx])
#         diffraction_image = np.array(Image.open(diffraction_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
#         diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0)

#         # Load corresponding original image (phase object)
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         image = np.array(Image.open(image_path).convert('L').resize((512, 512))) / 255.0  # 修改为512x512
#         image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

#         return diffraction_tensor, image_tensor  # Return both tensors

# # Main program
# image_dir = 'origin_image'  # Set your image directory
# diffraction_dir = 'dif_image'  # Set your diffraction directory
# dataset = PhaseObjectDataset(image_dir, diffraction_dir)

# # Split the dataset
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# # Create data loaders
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # Initialize the StarNet model
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 使用支持512x512的StarNet版本
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 200

# # Record losses
# train_losses = []
# test_losses = []

# # Training function
# def train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
#     best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
#     best_model_path = 'best_model_starnet.pth'  # 最佳模型权重保存路径

#     for epoch in range(num_epochs):
#         start_time = time.time()
#         model.train()
#         total_train_loss = 0.0

#         # Training phase
#         for data, targets in train_dataloader:
#             data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, targets)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)

#         # Testing phase
#         model.eval()
#         total_test_loss = 0.0
#         with torch.no_grad():
#             for data, targets in test_dataloader:
#                 data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到指定设备
#                 output = model(data)
#                 loss = criterion(output, targets)
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)

#         # 保存最佳模型权重
#         if average_test_loss < best_test_loss:
#             best_test_loss = average_test_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"New best model saved at epoch {epoch + 1} with Test Loss: {average_test_loss:.12f}")

#         elapsed_time = time.time() - start_time
#         print(f'Epoch [{epoch + 1}/{num_epochs}], '
#               f'Train Loss: {average_train_loss:.12f}, '
#               f'Test Loss: {average_test_loss:.12f}, '
#               f'Time: {elapsed_time / 60:.2f} minutes')

#     print(f"Training completed. Best model saved with Test Loss: {best_test_loss:.12f}")

# # Train the network
# train_network(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)

# # Save model weights
# torch.save(model.state_dict(), 'best_model_starnet.pth')

# # Plot loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='orange', linestyle='-')
# plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='blue', linestyle='-')
# plt.title('Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()

# # 设置 x 轴和 y 轴的范围，让它们从 0 开始
# plt.xlim(0, num_epochs + 1)
# plt.ylim(0, max(max(train_losses), max(test_losses)))
# plt.show()

# # Save loss values to Excel
# loss_data = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Loss': train_losses, 'Test Loss': test_losses})
# loss_data.to_excel('loss_values.xlsx', index=False)  # 保存到Excel文件