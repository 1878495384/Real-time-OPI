import os
import numpy as np
from scipy.fft import fft2, ifft2
from PIL import Image
import matplotlib.pyplot as plt


# ====================== 构建 Fresnel 角谱传递函数 ======================
def build_fresnel_H(shape, z, wavelength):
    """
    和你前向代码一致的 Fresnel 角谱传递函数:
    H(fx, fy) = exp(-i*k*z) * exp(-i*pi*lambda*z*(fx^2 + fy^2))

    shape: (H, W)
    z: 传播距离 (m)
    wavelength: 波长 (m)
    """
    height, width = shape
    k = 2 * np.pi / wavelength

    # 频率坐标
    fx = np.fft.fftfreq(width) * width
    fy = np.fft.fftfreq(height) * height
    FX, FY = np.meshgrid(fx, fy)  # (H, W)

    H = np.exp(-1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return H


# ====================== 角谱迭代算法：恢复频域随机相位 ======================
def recover_random_phase_angular_spectrum(image, diffraction_image, z, wavelength,
                                          n_iter=100, verbose=True):
    """
    利用角谱迭代算法，从“原强度图 + 相位畸变图”恢复频域随机相位。

    已知:
        image(x,y)             : 原强度图 (物面)，作为物面振幅 |U0|
        diffraction_image(x,y) : 相位畸变图 (像面)，为 |Uz|
        z, wavelength          : 与生成相位畸变图时一致的传播参数

    算法在两平面间迭代:
        频域: |S| = |F0| (F0 为 image 的 FFT)
        像面: |Uz| = diffraction_image
        传播关系: Uz = ifft2(S * H)

    返回:
        random_phase_est (H, W): 估计的频域随机相位 (rad, [-pi, pi))
    """
    # 预处理输入
    image = np.array(image, dtype=np.float64)
    diffraction_image = np.array(diffraction_image, dtype=np.float64)

    if image.max() > 0:
        image = image / image.max()
    if diffraction_image.max() > 0:
        diffraction_image = diffraction_image / diffraction_image.max()

    height, width = image.shape

    # 角谱传递函数 H 及其共轭
    H = build_fresnel_H((height, width), z, wavelength)
    H_conj = np.conj(H)

    # 物面场 U0 & 频域 F0
    U0 = image              # 振幅 = image，相位 = 0
    F0 = fft2(U0)           # 原始频域
    B = np.abs(F0) + 1e-12  # 频域幅度约束，避免除零

    # 像面幅度约束 A_z
    A_z = diffraction_image  # = |Uz|

    # 初始化频域场 S: 幅度 = B，相位随机
    init_phase = np.random.rand(height, width) * 2 * np.pi
    S = B * np.exp(1j * init_phase)

    for it in range(n_iter):
        # ---------- 频域 -> 像面 ----------
        Uz = ifft2(S * H)

        # 像面幅度约束：替换幅度为 A_z，保留相位
        Uz_phase = np.angle(Uz)
        Uz_constrained = A_z * np.exp(1j * Uz_phase)

        # ---------- 像面 -> 频域 ----------
        S_back = fft2(Uz_constrained) * H_conj

        # 频域幅度约束：|S| = B，保留相位
        S = B * np.exp(1j * np.angle(S_back))

        if verbose and ((it + 1) % 10 == 0 or it == n_iter - 1):
            Uz_test = ifft2(S * H)
            A_test = np.abs(Uz_test)
            if A_test.max() > 0:
                A_test = A_test / A_test.max()
            mse = np.mean((A_test - A_z) ** 2)
            print(f"Iter {it+1}/{n_iter}, MSE at image plane: {mse:.6e}")

    # 迭代结束，S_est ≈ F0 * exp(j * random_phase_est)
    S_est = S
    random_phase_est = np.angle(S_est * np.conj(F0))  # 相位差

    return random_phase_est


# ====================== 主程序：单张图像求随机相位 ======================
def main():
    # -------- 参数设置（根据你的实际情况修改） --------
    z = 0.1                # 衍射距离 (m)，要与生成相位畸变图时一致
    wavelength = 632.8e-9  # 波长 (m)
    n_iter = 100           # 迭代次数

    # -------- 输入路径（修改成你的实际路径） --------
    ori_path = 'frame/ori2/image998.png'   # 原强度图（物面）
    dif_path = 'frame/dif2/image998.png'   # 相位畸变图（像面幅度）

    if not os.path.exists(ori_path):
        raise FileNotFoundError(f"原强度图不存在: {ori_path}")
    if not os.path.exists(dif_path):
        raise FileNotFoundError(f"相位畸变图不存在: {dif_path}")

    # 读原始强度图（灰度）
    ori_img = Image.open(ori_path).convert('L')
    ori_arr = np.array(ori_img, dtype=np.float64)

    # 读相位畸变图（灰度）
    dif_img = Image.open(dif_path).convert('L')
    # 如果尺寸不一致，则按原图尺寸缩放相位畸变图
    if dif_img.size != ori_img.size:
        dif_img = dif_img.resize(ori_img.size, Image.BICUBIC)
    dif_arr = np.array(dif_img, dtype=np.float64)

    # 归一化到 [0,1]
    if ori_arr.max() > 0:
        ori_arr = ori_arr / ori_arr.max()
    if dif_arr.max() > 0:
        dif_arr = dif_arr / dif_arr.max()

    # -------- 调用角谱迭代算法恢复随机相位 --------
    random_phase_est = recover_random_phase_angular_spectrum(
        ori_arr, dif_arr, z, wavelength, n_iter=n_iter, verbose=True
    )

    # -------- 可视化并保存结果 --------
    phase_norm = (random_phase_est + np.pi) / (2 * np.pi)  # [-pi,pi] -> [0,1]
    phase_uint8 = (phase_norm * 255).astype(np.uint8)

    # 显示
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(ori_arr, cmap='gray')
    plt.title('Ground-truth (Input)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dif_arr, cmap='gray')
    plt.title('Diffraction image(Known)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(phase_norm, cmap='gray')
    plt.title('Reconstruction random phase')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存随机相位图
    out_name = 'recovered_phase998.png'
    Image.fromarray(phase_uint8).save(out_name)
    print(f"恢复的随机相位图已保存到: {out_name}")


if __name__ == '__main__':
    main()

















# import torch
# import torch.fft
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# import os
# import time

# # ================= 1. 配置与参数初始化 =================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on device: {device}")

# # 物理参数 (完全对应 MATLAB)
# lambda_val = 632.8e-6  # 波长 mm
# d = 150                # 衍射距离 mm
# N = 512                # 像素数
# PIESIZE = 8e-3         # 像素大小 mm
# L = N * PIESIZE        # 长宽
# k = 2 * np.pi / lambda_val # 波矢
# step = 30             # 迭代次数

# # 图像路径 (替换为你自己的图片路径)
# img_path = 'recovered_images_starnet(16)22222/recovered_image1909.png' 
# save_path = 'recover_phase.png' # 结果保存路径

# # ================= 2. 数据读取与预处理 =================
# def load_image(path, size=288):
#     if not os.path.exists(path):
#         print(f"未找到 {path}, 生成测试图像...")
#         # 生成一个中心矩形孔测试图
#         img = np.zeros((size, size), dtype=np.uint8)
#         img[100:200, 100:200] = 255
#         Image.fromarray(img).save(path)
    
#     img_pil = Image.open(path).convert('L') # 转灰度
#     img_pil = img_pil.resize((size, size))  # 强制 Resize 到 N
    
#     # 转为 Tensor [0, 1]
#     transform = transforms.ToTensor()
#     # MATLAB: A0=im2double(...) -> [0, 1] double
#     img_tensor = transform(img_pil).to(device).squeeze(0) # [H, W]
#     return img_tensor.float()

# # 读取 A0
# A0 = load_image(img_path, size=N)

# # 初始化变量
# # MATLAB: A=ones(N,N); phasek=2*pi.*rand(N,N);
# A = torch.ones(N, N, device=device)
# phasek = torch.rand(N, N, device=device) * 2 * np.pi
# phasek1 = phasek.clone()

# # 梯度与损失记录
# gk = torch.zeros(N, N, device=device)
# loss_history = []
# minloss = 1.0

# # ================= 3. 频域初始化 (角谱传递函数) =================
# # MATLAB: fX=[0:fix(x/2),ceil(x/2)-1:-1:1]./L;
# # PyTorch 的 fftfreq 自动处理这种频率排列 (无需手动构造)
# fx = torch.fft.fftfreq(N, d=PIESIZE).to(device)
# fy = torch.fft.fftfreq(N, d=PIESIZE).to(device)
# # meshgrid indexing='ij' 对应矩阵坐标
# FY, FX = torch.meshgrid(fy, fx, indexing='ij')

# # 计算传递函数 H
# # MATLAB: f=fx.^2+fy.^2;
# q = FX**2 + FY**2

# # MATLAB: H=exp(1j*k*d.*sqrt(1-(lambda*lambda).*(f)));
# # 注意处理负数开根号 (隐逝波)
# term_inside = 1 - (lambda_val**2) * q
# term_inside = term_inside.type(torch.complex64) # 转复数以处理 sqrt(-1)
# root_term = torch.sqrt(term_inside)

# H = torch.exp(1j * k * d * root_term)
# HB = 1.0 / H

# # ================= 4. 开始迭代 =================
# print("开始迭代...")
# start_time = time.time()

# # 初始物面 Ei
# # MATLAB: Ei=A.*exp(1j.*phasek);
# # 注意：虽然 phasek 是 tensor，但 exp 需要 tensor 为复数或者明确的 math 操作
# # 这里直接构造复数场
# Ei = A * torch.exp(1j * phasek)

# final_faik = None # 用于存储结果

# for n in range(step):
#     # --- Step 1: 正向传播 (物 -> 像) ---
#     # MATLAB: EOO=ifft2((fft2(Ei)).*H);
#     EOO = torch.fft.ifft2(torch.fft.fft2(Ei) * H)
    
#     # 计算强度并归一化 (Intensity)
#     # MATLAB: AOO=abs(EOO).^2; AOO=AOO./max(max(AOO));
#     AOO = torch.abs(EOO)**2
#     if AOO.max() > 0:
#         AOO = AOO / AOO.max()
        
#     # --- Step 2: 像面约束 (替换振幅) ---
#     # MATLAB: EO=A0.*exp(1j.*angle(EOO));
#     EO = A0 * torch.exp(1j * torch.angle(EOO))
    
#     # --- Step 3: 反向传播 (像 -> 物) ---
#     # MATLAB: Eii=ifft2((fft2(EO)).*HB);
#     Eii = torch.fft.ifft2(torch.fft.fft2(EO) * HB)
    
#     # --- Step 4: 提取相位并归一化 ---
#     # MATLAB: faik=angle(Eii); faik=faik./max(max(faik));
#     faik = torch.angle(Eii)
#     if faik.max() != 0:
#         faik = faik / faik.max()
    
#     # --- Step 5: 加速策略 (核心逻辑) ---
#     # MATLAB: beitak=(phasek-phasek1);
#     beitak = phasek - phasek1
    
#     current_gk = faik - phasek
    
#     if n > 0:
#         # MATLAB: rk=sum((gk.*gk1),"all")/(sum((gk1.^2),"all"));
#         # 注意: 这里的 gk 对应 current_gk, gk1 对应上一轮的 gk
#         gk1 = gk # 上一轮的梯度
        
#         # 计算分子分母
#         num = torch.sum(current_gk * gk1)
#         den = torch.sum(gk1**2)
        
#         if den != 0:
#             rk = num / den
#         else:
#             rk = 0.0
            
#         # 更新相位
#         # MATLAB: phasek=faik+beitak*rk;
#         new_phase = faik + beitak * rk
        
#         # 更新状态变量
#         gk = current_gk
#         phasek1 = phasek
#         phasek = new_phase
        
#         # MATLAB: phasek=phasek./max(max(phasek));
#         if phasek.max() != 0:
#             phasek = phasek / phasek.max()
            
#     else:
#         # 第一轮 (n=0)
#         # MATLAB: gk=faik-phasek; phasek=faik;
#         gk = current_gk
#         phasek1 = phasek
#         phasek = faik
    
#     # 更新 Ei 供下一次迭代
#     # MATLAB: Ei=exp(1j*phasek);
#     Ei = torch.exp(1j * phasek)
    
#     # --- Step 6: 计算 Loss ---
#     # MATLAB: loss(n)=immse(A0,AOO);
#     # MSE between Target Amplitude(A0) and Recon Intensity(AOO)
#     # (遵循原代码逻辑，虽然物理上通常比较 abs(E) vs abs(E))
#     loss = torch.mean((A0 - AOO)**2)
#     loss_history.append(loss.item())
    
#     # 更新最小loss对应的结果
#     if loss.item() < minloss:
#         minloss = loss.item()
#         final_faik = faik.clone() # 保存最佳结果
        
#     if (n+1) % 10 == 0:
#         print(f"Step [{n+1}/{step}], Loss: {loss.item():.6f}")

# print(f"Time elapsed: {time.time() - start_time:.2f}s")

# # ================= 5. 结果展示与保存 =================
# # 转为 numpy
# A0_np = A0.detach().cpu().numpy()
# faik_np = final_faik.detach().cpu().numpy() if final_faik is not None else phasek.detach().cpu().numpy()

# # 绘图
# plt.figure(figsize=(10, 5))

# # 原图
# plt.subplot(1, 2, 1)
# plt.imshow(A0_np, cmap='gray')
# plt.title("Target (A0)")
# plt.axis('off')

# # 恢复的相位
# plt.subplot(1, 2, 2)
# plt.imshow(faik_np, cmap='gray')
# plt.title("Recovered Phase")
# plt.axis('off')

# # # Loss 曲线
# # plt.subplot(1, 3, 3)
# # plt.plot(loss_history)
# # plt.title("MSE Loss")
# # plt.yscale('log')
# # plt.grid(True)

# plt.show()

# # 保存恢复的图像
# # MATLAB: faik=im2uint8(faik); imwrite(...)
# # 归一化到 0-255
# faik_norm = (faik_np - faik_np.min()) / (faik_np.max() - faik_np.min() + 1e-8)
# faik_uint8 = (faik_norm * 255).astype(np.uint8)

# try:
#     Image.fromarray(faik_uint8).save(save_path)
#     print(f"Saved recovered phase to: {save_path}")
# except Exception as e:
#     print(f"Error saving image: {e}")









    